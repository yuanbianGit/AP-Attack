import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss


def do_train_att_1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_SL_meter = AverageMeter()
    loss_Contrast_meter = AverageMeter()
    loss_meter = AverageMeter()

    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_Contrast_meter.reset()
        loss_SL_meter.reset()

        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            img = img.to(device)
            target = vid.to(device)
            with amp.autocast(enabled=True):
                image_feature, text_feature,_,_,_ = model(img, target)
                # print(image_feature.size())
                # print(text_feature.size())
            #   torch.Size([64, 512])
            #   torch.Size([64, 512])
            loss_i2t = xent(image_feature, text_feature, target, target)
            loss_t2i = xent(text_feature, image_feature, target, target)

            # 创建 batch_size x batch_size 的距离矩阵
            # dist_matrix = torch.cdist(text_feature.float(), text_feature.float(), p=2)  # 使用 L2 距离
            #
            # # 找到与每个 text_feature 相同身份的 text_feature (负样本)
            # mask_pos = target.unsqueeze(1) == target.unsqueeze(0)  # 正样本的 mask
            # loss_self = cfg.SOLVER.STAGE1.FACTOR2 * torch.sum(dist_matrix * mask_pos.float())  # 只保留正样本的距离，使得正样本文本一致！
            #
            loss_contrast = cfg.SOLVER.STAGE1.FACTOR*(loss_i2t + loss_t2i)

            # loss = loss_contrast + loss_self
            loss = loss_contrast

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            # loss_SL_meter.update(loss_self.item(),img.shape[0])
            loss_meter.update(loss.item(), img.shape[0])
            loss_Contrast_meter.update(loss_contrast.item(),img.shape[0])
            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss_all: {:.3f},Loss_ct: {:.3f},Loss_sl: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage1),
                                    loss_meter.avg,loss_Contrast_meter.avg, loss_SL_meter.avg,scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.SOLVER.STAGE1.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.SOLVER.STAGE1.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
