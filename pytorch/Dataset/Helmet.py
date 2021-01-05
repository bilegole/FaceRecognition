from pytorch.Dataset import DataSetOrigin

import torchvision.transforms as transform


class HelmetDataset(DataSetOrigin):
    def GetTrainTransform(self):
        return transform.Compose([
            transform.RandomCrop(32,padding=4),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def GetTestTransform(self):
        return transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def GetTrainDataset(self):
        pass
