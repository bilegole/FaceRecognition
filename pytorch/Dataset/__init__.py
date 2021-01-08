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


class DataSetOrigin:
    def initTrainLoader(self):
        self.trainloader = self.GetTrainLoader()

    def initTestLoader(self):
        self.testloader = self.GetTestLoader()

    def GetTrainTransform(self):
        pass

    def GetTestTransform(self):
        pass

    def GetTrainDataset(self):
        pass

    def GetTestDataset(self):
        pass

    def GetTrainLoader(self):
        pass

    def GetTestLoader(self):
        pass
