import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transform
import uuid
from pytorch.CIFAR.utils import progress_bar
import copy

# from .utils import progress_bar
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
# print(device)
path = os.path.abspath('../../.cache')
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truch')
# from ..CIFAR import
'''
全连接网络，主要就是一层隐藏层。
'''


class GeneralNetwork(nn.Module):
    def __init__(self, cache_path=path, id=None):
        super(GeneralNetwork, self).__init__()
        self.id = self.generate_id(id)
        self.root_path = cache_path
        self.checkpoint = os.path.join(self.root_path, 'chcekpoint')

        self.init_layers()
        self.GetOptimizer()
        self.train_loader = self.GetTrainLoader()
        self.test_loader = self.GetTestLoader()
        self.criterion = self.GetCriterion()

        self.acc: float = 0.
        self.best_acc: float = 0.
        self.epoch: int = 0
        self.history = dict()
        self.load_checkpoint()

    def init_layers(self):
        pass

    def name(self):
        return "未起名"

    def description(self):
        return "无特殊描述"

    def the_back_up(self):
        return os.path.join(self.checkpoint, self.id.__str__() + '.pth')

    def generate_id(self, id):
        if not id:
            id = uuid.uuid1()
            print(f"未指定uuid，生成：{id}")
        elif not (isinstance(id, str) and len(id) == 36):
            print(f"指定id不合法。请重新输入")
            sys.exit(0)
        else:
            print(f"指定id为{id}")
        return id

    def forward(self, x: torch.Tensor):
        return x

    def load_checkpoint(self):
        if os.path.exists(self.the_back_up()):
            checkpoint = torch.load(self.the_back_up())
            try:
                self.load_state_dict(checkpoint['net'])
                self.best_acc = checkpoint['acc']
                self.epoch = checkpoint['epoch']
                self.history = checkpoint['history']
                print(f"当前存档位置为：{self.the_back_up()},读取数据。\n当前最佳acc为\t{self.best_acc}\n轮次为\t{self.epoch}")
            except:
                print("网络结构已变更，是否无视存档？")
                if input("是否退出？(yes/no)") == 'yes':
                    sys.exit(0)
        else:
            print(f"未发现checkpoint")

    def save_checkpoint(self):
        state = {
            'net': self.state_dict(),
            'acc': self.acc,
            'epoch': self.epoch,
            'id':self.id,
            'name':self.name(),
            'description':self.description(),
            'history':self.history
        }
        if not os.path.isdir(self.checkpoint):
            os.mkdir(self.checkpoint)
        torch.save(state, os.path.join(self.checkpoint, self.id.__str__()) + '.pth')

    def train_and_test(self, epoch_count=1):
        tmp = self.epoch + 1
        for epoch in range(tmp, tmp + epoch_count):
            self.epoch = epoch
            self.Train(epoch, tmp + epoch_count)
            self.Test(epoch, tmp + epoch_count)
            if self.scheduler:
                self.scheduler.step()

    def Train(self, epoch: int, epoch_max: int):
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_index, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            # print("一轮训练")
            progress_bar(batch_index, len(self.train_loader), ' Loss: %.3f | Acc: %.3f%% (%6d%6d) | %d/%d | '
                         % (train_loss / (batch_index + 1), 100. * correct / total, correct, total, epoch, epoch_max))
        self.train_loss = train_loss

    def Test(self, epoch: int, epoch_max: int):
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # print("一轮测试")
                progress_bar(batch_index, len(self.test_loader), " Loss: %.3f | Acc: %.3f%% (%6d%6d) | %d/%d | "
                             % (
                                 test_loss / (batch_index + 1), 100. * correct / total, correct, total, epoch,
                                 epoch_max))

        self.acc = 100. * correct / total
        self.history[epoch] = copy.deepcopy({
            'acc':self.acc,
            'best_acc':self.best_acc,
            'correct':correct,
            'total':total,
            'train_loss':self.train_loss,
            'test_loss':test_loss
        })
        if self.acc > self.best_acc:
            self.save_checkpoint()

    def GetTrainLoader(self):
        self.transform_train = transform.Compose([
            transform.RandomCrop(32, padding=4),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        trainset = torchvision.datasets.CIFAR10(
            root=path, train=True, download=True, transform=self.transform_train
        )
        trainloader = DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=2
        )
        return trainloader

    def GetTestLoader(self):
        self.transform_test = transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = torchvision.datasets.CIFAR10(
            root=path, train=False, download=True, transform=self.transform_test
        )
        testloader = DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )
        return testloader

    def GetOptimizer(self, lr=1, momentum=0.9, weight_decay=5e-4):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,
                               momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,T_max=200)
        # return [self.optimizer,self.scheduler]
    def GetCriterion(self):
        return nn.CrossEntropyLoss()


class DenseNet(GeneralNetwork):
    def __init__(self, id=None):
        super(DenseNet, self).__init__(id=id)

    def init_layers(self):
        self.hidden_layer = nn.Linear(32 * 32 * 3, 100)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        return self.hidden_layer(x)

class DenseNet_1(GeneralNetwork):
    def __init__(self,id=None):
        super(DenseNet_1, self).__init__(id=id)

    def init_layers(self):
        self.input_size = 32 * 32 * 3
        self.hidden_layer = nn.Linear(self.input_size, 100)
        self.output_layer = nn.Linear(100, 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 32*32*3)
        x = self.hidden_layer(x)
        return self.output_layer(x)

    def GetOptimizer(self, lr=1, momentum=0.9, weight_decay=5e-4):
        return torch.optim.SGD(self.parameters(), lr=lr,
                               momentum=momentum, weight_decay=weight_decay)

    def GetCriterion(self):
        return nn.CrossEntropyLoss()
