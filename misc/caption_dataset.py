import json
import os
import re
from collections import defaultdict

import torch
from PIL import Image
from PIL import ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import ImageFilter
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ps_train_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file))
        self.transform = transform

        self.person2text = defaultdict(list)
        person_id2idx = {}
        n = 0
        self.pairs = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            person_id = ann['id']
            if person_id not in person_id2idx.keys():
                person_id2idx[person_id] = n
                n += 1
            person_idx = person_id2idx[person_id]
            if 'captions_bt' not in ann:
                ann['captions_bt'] = [''] * len(ann['captions'])
            for caption, caption_bt in zip(ann['captions'], ann['captions_bt']):
                caption = pre_caption(caption, max_words)
                caption_bt = pre_caption(caption_bt, max_words)
                self.pairs.append((image_path, caption, caption_bt, person_idx))
                self.person2text[person_idx].append(caption)


    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        image_path, caption, caption_bt, person = self.pairs[index]

        image_pil = Image.open(image_path)
        image = self.transform(image_pil.convert('RGB'))
        return {
            'image': image,
            'caption': caption,
            'caption_bt': caption_bt,
            'id': person,
        }


class ps_eval_dataset(Dataset):
    def __init__(self, ann_root, image_root, transform, split, max_words=30):
        ann_file = os.path.join(ann_root, split + '_reid.json')
        anns = json.load(open(ann_file, 'r'))
        self.transform = transform

        self.text = []
        self.image = []
        self.txt2person = []
        self.img2person = []

        for ann in anns:
            image_path = os.path.join(image_root, ann['file_path'])
            self.image.append(image_path)

            person_id = ann['id']
            self.img2person.append(person_id)
            for caption in ann['captions']:
                self.text.append(pre_caption(caption, max_words))
                self.txt2person.append(person_id)

        self.txt2person = torch.tensor(self.txt2person, dtype=torch.long)
        self.img2person = torch.tensor(self.img2person, dtype=torch.long)

    def __len__(self):
        return len(self.image)

    def __getitem__(self, index):
        image_path = self.image[index]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image

# 返回预处理后的文本字符串。最终返回的字符串将是去除了标点符号、多余空格、换行符，且不超过指定最大单词数的文本。
def pre_caption(caption, max_words=50):
    # 将文本中的所有标点符号 ([.!\"()*#:;~]) 替换为空格 ' '，并将文本全部转换为小写
    caption = re.sub(
        r"([.!\"()*#:;~])",
        ' ',
        caption.lower(),
    )
    # 将文本中连续出现的多个空白字符（例如空格、制表符等）替换为单个空格
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    # 去除文本末尾的换行符 \n，以确保文本不会因为末尾的换行而出现不必要的空白行。
    caption = caption.rstrip('\n')
    # 去除文本前后两端的空格，以确保文本前后没有多余的空格。
    caption = caption.strip(' ')

    # truncate caption
    # 将处理后的文本按空格拆分为一个单词列表 caption_words，以便检查和处理单词的数量。
    caption_words = caption.split(' ')
    # 如果单词数量超过了 max_words，则截取前 max_words 个单词，并将它们重新组合成一个新的字符串，保留到 caption 变量中。
    if len(caption_words) > max_words:
        caption = ' '.join(caption_words[:max_words])

    return caption