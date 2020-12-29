import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torchvision
import torchvision.transforms as transform

# from pytorch.CIFAR.utils import progress_bar
# from .utils import progress_bar
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
# print(device)
path = '../../.cache'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truch')
# from ..CIFAR import
'''
全连接网络，主要就是一层隐藏层。
'''

# _, term_width = os.popen('stty size', 'r').read().split()
# term_width = int(term_width)
term_width = 20
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
best_acc = 0
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

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
            # print("一轮训练")
            progress_bar(batch_index,len(train_loader),'Loss: %.3f | Acc: %.3f%% (%d%d)'
                               % (train_loss/(batch_index+1),100.*correct/total,correct,total))

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
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                # print("一轮测试")
                progress_bar(batch_index, len(test_loader),"Loss: %.3f | Acc: %.3f%% (%d%d)"
                             % (test_loss/(batch_index+1),100.*correct/total, correct,total))

        acc = 100. * correct / total
        # if acc > best_acc:
        #     print("此次运算不错，保存结果。")
        #     state = {
        #         "net": self.stat_dict(),
        #         "acc": acc,
        #         "epoch": epoch
        #     }
        #     if not os.path.isdir('checkpoint'):
        #         os.mkdir('checkpoint')
        #     torch.save(state, './checkpoint/model.pth')
        #     best_acc = acc

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
            root=path, train=False, download=True,transform=self.transform_test
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
        x = x.view(-1,32*32*3)
        return self.hidden_layer(x)


class SimpleDenseNetwork(GeneralNetwork):
    def __init__(self):
        super(SimpleDenseNetwork, self).__init__()


class SimpleCNN(GeneralNetwork):
    def __init__(self):
        super(SimpleCNN, self).__init__()

    def forward(self, x):
        pass
