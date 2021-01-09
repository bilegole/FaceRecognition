# Author: yuang
# @Time   : 2021/1/6 10:58
# @Author : yuang
# @Site   : www.bilegole.com
# @File   : VOC.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import torchvision
import torchvision.transforms as transform
import torchvision.datasets as datasets

from pytorch.config import cache_path
from pytorch.Dataset import DataSetOrigin

import os, time, sys
import matplotlib.pyplot as plt
from typing import Tuple, TypeVar, List, Dict
import PIL.Image as Img

Image = Img.Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


# TODO:完成 VOCDataSet.
class VOCDataSet(DataSetOrigin):
    def __init__(self, year: str = '2012'):
        super(VOCDataSet, self).__init__()
        self.data_loaded = True
        self.year = year
        self.SizeOfPictureByPixel = 448

        self.Classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                        "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
                        "tvmonitor"]
        print('VOCDataset 初始化完成.')

    def GetTrainTransform(self):
        return transform.Compose([
            transform.ToTensor()
        ])

    def GetTrainDataset(self):
        return datasets.VOCDetection(
            cache_path,
            year=self.year,
            image_set='train',
            download=False,
            # transform=self.GetTrainTransform()
        )

    def GetTrainLoader(self):
        return DataLoader(self.GetTrainDataset(), batch_size=10, shuffle=False, collate_fn=self.GetTrainCollectFun)

    def GetTrainCollectFun(self, batch: List):
        # print(1)
        # print(batch)
        images = []
        labels = []
        for index, (image, target) in enumerate(batch):
            index: torch.Tensor
            image: Image
            target: Dict

            image_size: Tuple[int, int] = image.size
            image_to_scale = (self.SizeOfPictureByPixel / i for i in image_size)

            _image = image.resize((self.SizeOfPictureByPixel, self.SizeOfPictureByPixel))
            trans = self.GetTrainTransform()
            image_tensor = trans(_image)
            images.append(image_tensor)

            objects = target['annotation']['object']
            label_tensor = FloatTensor(len(objects), 6).fill_(0)
            for i, object in enumerate(objects):
                object: Dict
                xmin = int(object['bndbox']['xmin'])
                xmax = int(object['bndbox']['xmax'])
                ymin = int(object['bndbox']['ymin'])
                ymax = int(object['bndbox']['ymax'])
                label = object['name']

                label_num = self.Classes.index(label)
                # x,y是方框中心的横纵坐标相对于整个图片边长的比值。
                x: float = (xmin + xmax) / (2 * image_size[0])
                y: float = (ymin + ymax) / (2 * image_size[1])
                # wh是方框的宽度和高度相对于图片边长的比值。
                w: float = (xmax - xmin) / image_size[0]
                h: float = (ymax - ymin) / image_size[1]  #

                label_tensor[i, :] = torch.tensor([index, label_num, x, y, w, h])
            labels.append(label_tensor)

        # _images = torch.stack(images, dim=0)
        _labels = torch.cat(labels, dim=0)
        _images = torch.stack(images, dim=0)
        return _images, _labels

    def GetTestTransform(self):
        return transform.Compose([
            transform.ToTensor()
        ])

    def GetTestDataset(self):
        return datasets.VOCDetection(
            cache_path,
            year=self.year,
            image_set='val',
            download=False,
            # transform=self.GetTestTransform()
        )

    def GetTestLoader(self):
        return DataLoader(self.GetTestDataset(), batch_size=32, shuffle=False, collate_fn=self.GetTrainCollectFun)

    def TestCollectFun(self, batch: Tuple):
        images, labels = batch
        image: Image
        labels: Dict

        pass


if __name__ == '__main__':
    a = VOCDataSet()
    loader = a.GetTrainLoader()
    b = [i for i in loader]
    print(b[0])
