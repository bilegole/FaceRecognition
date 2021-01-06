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

import os, time, sys
import matplotlib.pyplot as plt

from pytorch.config import cache_path
from pytorch.Dataset import DataSetOrigin

device = "cuda:0" if torch.cuda.is_available() else "cpu"


# TODO:完成 VOCDataSet.
class VOCDataSet(DataSetOrigin):
    def __init__(self, year: str = '2007'):
        super(VOCDataSet, self).__init__()
        self.year = year

    def GetTrainTransform(self):
        pass

    def GetTrainDataset(self):
        return datasets.VOCDetection(
            cache_path,
            year=self.year,
            image_set='train',
            download=True,
            transform=self.GetTrainTransform()
        )

    def GetTrainLoader(self):
        pass

    def GetTestTransform(self):
        pass

    def GetTestDataset(self):
        return datasets.VOCDetection(
            cache_path,
            year=self.year,
            image_set='test',
            download=True,
            transform=self.GetTestTransform()
        )

    def GetTestLoader(self):
        DataLoader(self.GetTestLoader(),batch_size=1000,shuffle=True)

