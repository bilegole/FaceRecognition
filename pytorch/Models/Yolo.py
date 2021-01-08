# Author: yuang
# @Time   : 2021/1/8 15:08
# @Author : yuang
# @Site   : www.bilegole.com
# @File   : Yolo.py.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transform
import torchvision.datasets as datasets

import os, time, sys
import matplotlib.pyplot as plt

from pytorch.config import cache_path
from pytorch.Models import GeneralNetwork

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class Yolo_v1(GeneralNetwork):
    def init_layers(self):
        self.Seq_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=5),
            nn.MaxPool2d(2, stride=2)
        )
        self.Seq_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3),
            nn.MaxPool2d(2, stride=2)
        )
        self.Seq_3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3),
            nn.MaxPool2d(2, stride=2)
        )
        self.subSeq_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        )
        self.Seq_4 = nn.Sequential(
            self.subSeq_4, self.subSeq_4, self.subSeq_4, self.subSeq_4,
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3),
            nn.MaxPool2d(2, stride=2)
        )
        self.subSeq_5 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        )
        self.Seq_5 = nn.Sequential(
            self.subSeq_5, self.subSeq_5,
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=2)
        )
        self.Seq_6 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)
        )
        self.fc_1 = nn.Linear(in_features=1024, out_features=4096)
        self.fc_2 = nn.Linear(in_features=4096, out_features=30)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1047)
        )

    def forward(self, x):
        x = self.Seq_1(x)
        x = self.Seq_2(x)
        x = self.Seq_3(x)
        x = self.Seq_4(x)
        x = self.Seq_5(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        x = F.sigmoid(x)
        x = x.view(-1,7,7,30)
        return x
