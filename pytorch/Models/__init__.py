import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import torchvision
import torchvision.transforms as transforms

from pytorch.common.utils import check_id

import os, sys, copy

cache_path = os.path.abspath('E:\\.cache')
device = 'cuda:0' if torch.cuda.is_available() else "cpu"


class GeneralNetwork(nn.Module):
    def __init__(self, model_id=None, cache_path: str = cache_path, dry: bool = False):
        super(GeneralNetwork, self).__init__()
        self.model_loaded = True
        self.id = self.check_id(model_id)
        self.cache_path = cache_path
        self.base_path = os.path.join(self.cache_path, self.dir_name())
        self.checkpoint = os.path.join(self.base_path, 'checkpoint')
        self.dry: bool = dry
        if not os.path.exists(self.base_path):
            os.mkdir(self.base_path)
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)

        self.init_layers()
        self.optimizer, self.scheduler = self.GetOptimizer()
        self.criterion = self.GetCriterion()
        self.trainloader: DataLoader
        self.testloader: DataLoader

        self.acc: float = 0.
        self.best_acc: float = 0.
        self.epoch: int = 0
        self.history = dict()

        self.load_checkpoint()

    def dir_name(self):
        return 'default'

    def init_layers(self):
        pass

    def GetOptimizer(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.0005, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        return optimizer, scheduler

    def GetCriterion(self):
        return nn.CrossEntropyLoss()

    def name(self):
        return self.__class__.__name__
        pass

    def check_id(self, model_id):
        return check_id(model_id)
        pass

    def description(self):
        return "unKnown"

    def the_back_up(self):
        return os.path.join(self.checkpoint, self.id.__str__() + '.pth')

    def save_checkpoint(self):
        state = {
            'net': self.state_dict(),
            'acc': self.acc,
            'epoch': self.epoch,
            'id': self.id,
            'name': self.name(),
            'description': self.description(),
            'history': self.history
        }
        if not os.path.isdir(self.checkpoint):
            os.mkdir(self.checkpoint)
        torch.save(state, self.the_back_up())

    def load_checkpoint(self):
        if os.path.exists(self.the_back_up()):
            checkpoint = torch.load(self.the_back_up())
            try:
                self.load_state_dict(checkpoint['net'])
            except:
                if input("网络结构已更新,是否覆盖存档?(yes/no)") == 'no':
                    sys.exit(0)
            self.best_acc = checkpoint['acc']
            self.epoch = checkpoint['epoch']
            self.history = checkpoint['history']
            print(f"当前存档位置为{self.the_back_up()},读取数据...\n当前最佳acc为\t{self.best_acc}\n轮次为\t{self.epoch}")
        else:
            print(f'在 "{self.the_back_up()}"\n未发现checkpoint')
        pass

    def train_and_test(self, epoch_count: int = 1):
        if not hasattr(self, 'data_loaded'):
            raise Exception("未导入数据.")
        self.initTrainLoader()
        self.initTestLoader()
        start = self.epoch + 1
        end = self.epoch + epoch_count
        for epoch in range(start, end + 1):
            self.epoch = epoch
            self.Train()
            self.Test()
            if hasattr(self, 'scheduler') and self.scheduler:
                self.scheduler.step()
            self.displayEpochStatus()
            if self.dry == True:
                break

    def Train(self):
        self.train_loss_total = 0
        self.train()
        self.max_index = len(self.trainloader)
        for batch_index, (inputs, targets) in enumerate(self.trainloader):
            self.curr_index = batch_index
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.displayTrainStatus(inputs=inputs, outputs=outputs, targets=targets, loss=loss)
            if self.dry:
                break

    def displayTrainStatus(self, inputs, outputs, targets, loss):
        pass

    def Test(self):
        self.eval()
        self.test_loss_total = 0
        with torch.no_grad():
            self.max_index = len(self.testloader)
            for batch_index, (inputs, targets) in enumerate(self.testloader):
                self.curr_index = batch_index
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)
                self.displayTestStatus(inputs, outputs, targets, loss)
                if self.dry:
                    break

    def displayTestStatus(self, inputs, outputs, targets, loss):
        pass

    def displayEpochStatus(self):
        pass

    def forward(self, x):
        return x
