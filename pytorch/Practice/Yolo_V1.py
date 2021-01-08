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

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class YoloV1(Yolo_v1, VOCDataSet):
    def __init__(self):
        Yolo_v1.__init__(self)
        VOCDataSet.__init__(self)

    def dir_name(self):
        return 'test'

    pass


if __name__ == '__main__':
    net = YoloV1().to(device)
    print('End')
    loader = net.GetTrainLoader()
    datas = [i for i in loader]
    input, target = datas[0]
    del loader
    del datas
    input = input.to(device)
    x1 = net.Seq_1(input)
    x2 = net.Seq_2(x1)
    x3 = net.Seq_3(x2)
    x4 = net.Seq_4(x3)
    x5 = net.Seq_5(x4)
    output = net(input)