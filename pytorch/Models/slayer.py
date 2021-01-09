# Author: yuang
# @Time   : 2021/1/8 15:12
# @Author : yuang
# @Site   : www.bilegole.com
# @File   : slayer.py

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class slayer():
    pass

