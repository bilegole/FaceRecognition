import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import torchvision.transforms as transform
from torchvision.datasets.vision import VisionDataset

from pytorch.Dataset import DataSetOrigin
from pytorch.config import cache_path

# import

# TODO:完成HelmetDataset部分代码。
class HelmetDataset(VisionDataset):
    def __init__(self, root: str = cache_path, transform=None, transforms=None, target_transform=None):
        super(HelmetDataset, self).__init__(
            root=root,
            transform=transforms
        )


class HelmetDataSet(DataSetOrigin):
    def GetTrainTransform(self):
        return transform.Compose([
            transform.RandomCrop(32, padding=4),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def GetTestTransform(self):
        return transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def GetTrainDataset(self):
        pass

    def GetTestDataset(self):
        pass
