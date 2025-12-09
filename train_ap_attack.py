import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from utils.logger import setup_logger
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from utils.meter import AverageMeter

from loss.make_loss import make_loss
import torch.nn as nn
import random
import torch
import numpy as np
import argparse
from config import cfg
from advers.GD import Generator, MS_Discriminator, Pat_Discriminator, GANLoss, weights_init,ResnetG
import torch.optim as optim
from torch.cuda import amp
from utils.metrics import R1_mAP_eval_att
import errno
import os.path as osp
import torch.nn.functional as F
from torchvision import transforms as transforms_torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Imagenet_mean = [0.485, 0.456, 0.406]
# Imagenet_stddev = [0.229, 0.224, 0.225]
Imagenet_mean = [0.5, 0.5, 0.5]
Imagenet_stddev = [0.5, 0.5, 0.5]

inv_mean = [-m / s for m, s in zip(Imagenet_mean, Imagenet_stddev)]
inv_stddev = [1 / s for s in Imagenet_stddev]

Imagenet_mean_IDE = [0.485, 0.456, 0.406]
Imagenet_stddev_IDE = [0.229, 0.224, 0.225]

trans_G = transforms_torch.Compose([transforms_torch.Normalize(inv_mean,inv_stddev),
                              transforms_torch.Normalize(mean=Imagenet_mean_IDE,std=Imagenet_stddev_IDE)])

import torchvision
from torch.nn import init

class IDE(nn.Module):
    def __init__(self, pretrained=True, cut_at_pooling=False,
                 num_features=1024, norm=False, dropout=0, num_classes=0):
        super(IDE, self).__init__()

        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        # Construct base (pretrained) resnet
        self.base = torchvision.models.resnet50(pretrained=True)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal(self.feat.weight, mode='fan_out')
                init.constant(self.feat.bias, 0)
                init.constant(self.feat_bn.weight, 1)
                init.constant(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal(self.classifier.weight, std=0.001)
                init.constant(self.classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, is_training=False, metaTrain=True, mix_thre=0.6,mix_pro= 0.5, output_both=False, mix_info=None, lamd=None):

        x = self.base.conv1(x)

        x = self.base.bn1(x)
        x = self.base.relu(x)
        # print(randE)

        x_layer0 = self.base.maxpool(x)
        x_layer1 = self.base.layer1(x_layer0)
        x_layer2 = self.base.layer2(x_layer1)
        x_layer3 = self.base.layer3(x_layer2)
        feat_map = self.base.layer4(x_layer3)
        x = feat_map

        # part_feats = nn.AdaptiveAvgPool2d((6,1))(feat_map).squeeze()
        # part_feats =part_feats.view(part_feats.size(0), -1)

        if self.cut_at_pooling:
            return x
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if self.has_embedding:
            x = self.feat(x)
            x = self.feat_bn(x)

        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            logits = self.classifier(x)

        if is_training:
            return logits, x
        else:
            return x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

class adv_TripletLoss(nn.Module):
  def __init__(self, margin=0.3):
    super(adv_TripletLoss, self).__init__()
    self.margin = margin

  def forward(self, features_clean, features_adv, pids):
       """
        反向三元组损失计算，找到最不相似的负样本进行拉近。
        :param features_clean: 干净的 features, 形状为 (batch_size, 1024)
        :param features_adv: 对抗的 features, 形状为 (batch_size, 1024)
        :param pids: 每个样本的身份标签 (person IDs), 形状为 (batch_size,)
        :param margin: margin，控制推远的程度，默认为 1.0
        :return: loss 值
      """
       features_clean = F.normalize(features_clean, p=2, dim=1)
       features_adv = F.normalize(features_adv, p=2, dim=1)

       # 计算每个对 (features_adv, features_clean) 的欧氏距离
       dist_adv = torch.cdist(features_adv,features_clean , p=2)

       # 创建 batch_size x batch_size 的距离矩阵，算干净样本的距离矩阵，找出最远的样本
       dist_matrix = torch.cdist(features_clean, features_clean, p=2)  # 使用 L2 距离

       # 找到与每个 feature_adv 不同身份的 feature_clean (负样本)
       mask_neg = pids.unsqueeze(1) != pids.unsqueeze(0)  # 负样本的 mask
       dist_neg = dist_matrix * mask_neg.float()  # 只保留负样本的距离
       _, hard_neg_indices = torch.max(dist_neg, dim=1)
       _, hard_pos_indices = torch.min(dist_matrix, dim=1)

       dist_neg_max = dist_adv.gather(1, hard_neg_indices.unsqueeze(1)).squeeze(1)
       dist_pos = dist_adv.gather(1, hard_pos_indices.unsqueeze(1)).squeeze(1)

       loss = F.relu( dist_neg_max-dist_pos + self.margin)

       return loss.mean()

def perturb_train(imgs, G, train_or_test='test',cam_ids=None,randE=None):
  # print('----------'+imgs.type())
  delta= G(imgs)
  # logger.info(delta.size())
  # NOTE 这里的L_norm能保障灰度图像加上还是灰度图像
  delta = L_norm(delta,train_or_test)
  # logger.info(delta.size()) torch.Size([128, 1, 288, 144])
  # new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())
  # 加入噪声和没加入噪声的!
  # _, mask = D(torch.cat((imgs, new_imgs.detach()), 1))
  #
  # delta = delta * mask
  mask = torch.ones_like(delta).cuda()
  # delta = torch.zeros_like(delta).cuda()
  new_imgs = torch.add(imgs.cuda(), delta[0:imgs.size(0)].cuda())
  for c in range(3):
    new_imgs.data[:,c,:,:] = new_imgs.data[:,c,:,:].clamp((0.0 - Imagenet_mean[c]) / Imagenet_stddev[c],
                                                          (1.0 - Imagenet_mean[c]) / Imagenet_stddev[c]) # do clamping per channel

  if train_or_test == 'train':
    return new_imgs, mask
  elif train_or_test == 'test':
    return new_imgs, delta, mask

def L_norm(delta, mode='train'):
  '''
  大概意思就是clamp使得噪声符合约束！
  '''
  # 当时的transform ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
  delta = torch.clamp(delta,-8.0/255,8.0/255)
  if delta.size(1)==1:
      delta = delta.repeat(1,3,1,1)
  for c in range(delta.size(1)):
    delta.data[:,c,:,:] = (delta.data[:,c,:,:]) / Imagenet_stddev[c]
  return delta

pdist = torch.nn.PairwiseDistance(p=2)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, is_best, G_or_D, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    # torch.save(state, fpath)
    if is_best:
        # shutil.copy(fpath, osp.join(osp.dirname(fpath), 'best_'+ G_or_D +'.pth.tar'))
        torch.save(state, osp.join(osp.dirname(fpath), 'best_'+ G_or_D +'.pth.tar'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/att/vit_clipreid_att.yml", help="path to config file", type=str
    )

    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument("--local_rank", default=0, type=int)
    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    if cfg.MODEL.DIST_TRAIN:
        torch.cuda.set_device(args.local_rank)
    adv_tri = adv_TripletLoss(margin=cfg.SOLVER.STAGE2.ADV_FACTOR_3_MARGIN)
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DIST_TRAIN:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    train_loader_stage2, train_loader_stage1, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    device = "cuda"

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    # NOTE 这里导入最后的模型
    model.load_param(cfg.SOLVER.STAGE2.TARGET_MODEL)
    model.eval()

    surrrgot_model = IDE(num_classes=num_classes).cuda()
    surrrgot_model.load_param('./ide_dukemtmcreid.pth')

    surrrgot_model.eval()

    G_V = Generator(3, 3, 32, norm='bn', beta=0.1).apply(weights_init).to(device)

    optimizer_G_V = optim.Adam(G_V.parameters(), lr=cfg.SOLVER.STAGE2.BASE_LR, betas=(0.5, 0.999))
    # optimizer_D_V = optim.Adam(D_V.parameters(), lr=cfg.SOLVER.STAGE2.BASE_LR, betas=(0.5, 0.999))

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    criterionGAN = GANLoss()
    save_dir = cfg.OUTPUT_DIR
    scaler_G_V = amp.GradScaler()
    scaler_D_V = amp.GradScaler()

    epochs =cfg.SOLVER.STAGE2.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    best_hit, best_epoch = np.inf, 0
    model.to(device)

    loss_meter = AverageMeter()

    loss_1_meter = AverageMeter()
    loss_2_meter = AverageMeter()
    loss_3_meter = AverageMeter()

    for epoch in range(1, epochs + 1):
        model.eval()

        G_V.train()
        # D_V.train()
        loss_meter.reset()
        loss_1_meter.reset()
        loss_2_meter.reset()
        loss_3_meter.reset()

        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):

            num = cfg.SOLVER.STAGE2.IMS_PER_BATCH // 2

            with amp.autocast(enabled=True):
                img = img.to(device)
                target = vid.to(device)
                img_adv, mask_V = perturb_train(img, G_V, train_or_test='train')

                img_f_clip, text_feature,image_feature,_, prompt_clean = model(img, target)
                img_f_clip_adv, text_feature_adv,image_feature_adv,_,prompt_adv  = model(img_adv, target)

                surr_feat = surrrgot_model(trans_G(img))
                surr_feat_adv = surrrgot_model(trans_G(img_adv))


                loss_G_ReID_V_Surr = cfg.SOLVER.STAGE2.ADV_FACTOR * adv_tri(surr_feat.detach(), surr_feat_adv, target)



                semantic_clean_text = torch.chunk(prompt_clean, 5, dim=0)
                semantic_adv_text = torch.chunk(prompt_adv, 5, dim=0)

                # NOTE 这是最终的loss, fine grained text
                loss_G_ReID_V_text_Fine = cfg.SOLVER.STAGE2.ADV_FACTOR_3 * torch.sum(torch.cat(
                    [adv_tri(semantic_clean_text[i].detach(), semantic_adv_text[i], target).unsqueeze(0) for i in
                     range(len(semantic_adv_text))]))

                loss_G_ReID_V = loss_G_ReID_V_Surr + loss_G_ReID_V_text_Fine


                loss_meter.update(loss_G_ReID_V.item(),img.shape[0])
                loss_1_meter.update(loss_G_ReID_V_Surr.item(),img.shape[0])
                loss_3_meter.update(loss_G_ReID_V_text_Fine.item(),img.shape[0])

            optimizer_G_V.zero_grad()
            scaler_G_V.scale(loss_G_ReID_V).backward()
            scaler_G_V.step(optimizer_G_V)
            scaler_G_V.update()



            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:

                logger.info(
                    "===> Epoch[{}]({}/{}) loss_G_ReID_V: {:.2f} loss_G_ReID_V_Surr: {:.2f} loss_G_ReID_V_text_Fine: {:.2f}".format(
                        epoch, n_iter+1, len(train_loader_stage2),loss_meter.avg, loss_1_meter.avg, loss_3_meter.avg,
                        ))

        if epoch % cfg.SOLVER.STAGE2.EVAL_PERIOD == 0:
            evaluator_adv = R1_mAP_eval_att(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
            evaluator_adv.reset()
            G_V.eval()
            model.eval()
            surrrgot_model.eval()
            for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)

                    with amp.autocast(enabled=True):
                        img_adv, mask_V = perturb_train(img, G_V, train_or_test='train')
                        target = pid.to(device)
                        feat= surrrgot_model(trans_G(img))
                        feat_att= surrrgot_model(trans_G(img_adv))

                        evaluator_adv.update((feat.float(),feat_att.float(), pid, camid))

            cmc, mAP, cmc_att, mAP_att= evaluator_adv.compute()
            logger.info("Validation Results ")
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

            logger.info("Validation Results ")
            logger.info("mAP_att: {:.1%}".format(mAP_att))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc_att[r - 1]))
            is_best = mAP_att <= best_hit
            if is_best:
                best_hit, best_epoch = mAP_att, epoch

            logger.info("==> Best_epoch is {}, Best rank-1 {:.1%}".format(best_epoch, best_hit))
            save_checkpoint(G_V.state_dict(), is_best, 'G_V', osp.join(save_dir, 'G_V_ep' + str(epoch) + '.pth.tar'))




