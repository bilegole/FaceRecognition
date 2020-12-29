import os

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transform

device = 'cuda:0' if torch.cuda.is_available() else "cpu"
print(device)
path = '../../.cache'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truch')
# from ..CIFAR import
'''
全连接网络，主要就是一层隐藏层。
'''


class GeneralNetwork(nn.Module):
    def __init__(self):
        super(GeneralNetwork, self).__init__()

        pass

    def forward(self, x):
        pass

    def train_and_test(self, start_epoch=0):
        for epoch in range(start_epoch, start_epoch + 200):
            self.Train(epoch)
            self.Test(epoch)

    def TheOptimizer(self):
        pass

    def Train(self, epoch: int):
        self.train()
        train_loss = 0
        correct = 0
        total = 0
        train_loader = self.GetTrainLoader()
        optimizer = self.GetOptimizer()
        criterion = self.GetCriterion()
        for batch_index, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print("此处还需要添加进度条。")

    def Test(self, epoch: int):
        global best_acc
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        test_loader = self.GetTestLoader()
        criterion = self.GetCriterion()
        with torch.no_grad():
            for batch_index, (inputs, targets) in enumerate(test_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.max(1)
                correct += predicted.eq(targets).sum().item()

            print("此处需要添加进度条。")
        acc = 100. * correct / total
        if acc > best_acc:
            print("此次运算不错，保存结果。")
            state = {
                "net": self.stat_dict(),
                "acc": acc,
                "epoch": epoch
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/model.pth')
            best_acc = acc

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
            root=path, train=False, download=True
        )
        testloader = DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2
        )
        return testloader

    def GetOptimizer(self, lr=1, momentum=0.9, weight_decay=5e-4):
        return torch.optim.SGD(self.parameters(), lr=lr,
                               momentum=momentum, weight_decay=weight_decay)

    def GetCriterion(self):
        return nn.CrossEntropyLoss()


class DenseNet(GeneralNetwork):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.hidden_layer = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        return self.hidden_layer(x)


class SimpleCNN(GeneralNetwork):
    def __init__(self):
        super(SimpleCNN, self).__init__()

    def forward(self, x):
        pass
