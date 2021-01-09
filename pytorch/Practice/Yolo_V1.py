# Author: yuang
# @Time   : 2021/1/8 19:28
# @Author : yuang
# @Site   : www.bilegole.com
# @File   : Yolo_V1.py

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
from pytorch.Models.Yolo import Yolo_v1
from pytorch.Dataset.VOC import VOCDataSet
from pytorch.Loss.YoloLoss import YoloLoss_v1

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloV1(Yolo_v1, VOCDataSet):
    def __init__(self, dry: bool = False, batch_size: int = 10, test_batch_size: int = 16):
        Yolo_v1.__init__(self, dry=dry, )
        VOCDataSet.__init__(self, batch_size=batch_size)
        # self.: int = batch_size

    def dir_name(self):
        return 'test'

    def GetCriterion(self):
        # batch_size = self.batch_size
        return YoloLoss_v1(num_sample=10)

    def GetOptimizer(self):
        return optim.SGD(self.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4),None

    # def displayTrainStatus(self, inputs, outputs, targets, loss):
    #     print('完成了一轮Train')

    # def displayTestStatus(self, inputs, outputs, targets, loss):
    #     print('完成了一轮Test')

    def displayEpochStatus(self):
        print('完成了一轮epoch')


if __name__ == '__main__':
    # net = YoloV1().to(device)
    # print('End')
    # loader = net.GetTrainLoader()
    # datas = [i for i in loader]
    # input, target = datas[0]
    # del loader
    # del datas
    # input = input.to(device)
    # x1 = net.Seq_1(input)
    # x2 = net.Seq_2(x1)
    # x3 = net.Seq_3(x2)
    # x4 = net.Seq_4(x3)
    # x5 = net.Seq_5(x4)
    # output = net(input)
    net = YoloV1(dry=True, batch_size=20).to(device)
    net.train_and_test(epoch_count=100)
