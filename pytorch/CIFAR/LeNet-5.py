from pytorch.CIFAR import model
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(model.GeneralNetwork):
    def init_layers(self):
        self.inchannel = 3
        self.outtype = 10
        self.channel_1 = 6
        self.channel_2 = 16
        self.conv_1 = nn.Conv2d(self.inchannel,self.channel_1,5)
        self.conv_2 = nn.Conv2d(self.channel_1,self.channel_2,5)
        self.fc_size = [self.channel_2*5*5,120,84,self.outtype]
        self.fc1 = nn.Linear(self.fc_size[0],self.fc_size[1])
        self.fc2 = nn.Linear(self.fc_size[1],self.fc_size[2])
        self.fc3 = nn.Linear(self.fc_size[2],self.fc_size[3])

    def forward(self, x: torch.Tensor):
        x = F.relu(self.conv_1(x))
        x = F.max_pool2d(x,2)
        x = F.relu(self.conv_2(x))
        x = F.max_pool2d(x,2)
        x = x.view(-1,16*5*5)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def GetOptimizer(self, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr,
                                         momentum=momentum, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)



if __name__ == '__main__':
    print('开始初始化网络。。。')
    net = LeNet(id="543e3a1d-4b31-11eb-82c5-a4b1c131cc51").to(model.device)
    print("网络初始化完成，开始训练")
    net.train_and_test(epoch_count=20)
    print("网络训练完成")
