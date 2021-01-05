import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transform

import os, time, sys


class DataSetOrigin:
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
