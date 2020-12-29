import torch
import torch.nn as nn

import pytorch.helloworld.common as common

OUTPUT_SIZE = 10
device = "cuda:0" if torch.cuda.is_available() else "cpu"

class ConvNetSimple(nn.Module):
    def __init__(self):
        super(ConvNetSimple, self).__init__()
        self.conv_1_size = 12
        self.out_channels = 20
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5)
        self.max_pool_2d = lambda x: torch.max_pool2d(x, kernel_size=2, stride=2)
        self.view = lambda x: x.view(-1, self.conv_1_size ** 2 * 2 * 20)
        self.fc_1 = nn.Linear(self.conv_1_size ** 2 * self.out_channels, 100)
        self.out = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        # layers = [
        #     ["conv_1", self.conv_1],
        #     ["sigmoid", torch.sigmoid],
        #     ["max_pool", self.max_pool_2d],
        #     ["flatten", self.view],
        #     ["full_connect", self.fc_1],
        #     ["final_sigmoid", torch.sigmoid],
        #     ["out", self.out],
        # ]
        # for (layername,layer) in layers:
        #     x = layer(x)
        # return x
        x = torch.sigmoid(self.conv_1(x))
        x = torch.max_pool2d(x,kernel_size=2,stride=2)

        x = x.view(-1,self.conv_1_size**2*self.out_channels)
        x = torch.sigmoid(self.fc_1(x))

        x = self.out(x)
        return x




def train_and_test_network(net,num_epochs=60,lr=0.1,wd=0,
                           loss_function=nn.CrossEntropyLoss().to(device),
                           train_loader=common.get_train_loader(),
                           test_loader=common.get_test_loader()):
    sgd = torch.optim.SGD(net.parameters(),lr=lr,weight_decay=wd)

    common.train_network(net,train_loader,num_epochs,loss_function,sgd)

    print("")

    network = common.test_network(net, train_loader)

if __name__ == '__main__':
    print(f"using device {device}")
    net = ConvNetSimple().to(device)
    train_and_test_network(net)
