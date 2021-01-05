from pytorch.CIFAR import model
import torch
import torch.nn as nn
import torch.nn.functional as F

Cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}
vgg_type = 'VGG11'


class VGG(model.GeneralNetwork):
    def init_layers(self):
        self.classifier = nn.Linear(512, 10)
        assert vgg_type in Cfg
        cfg = Cfg[vgg_type]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                x: int
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                    # nn.Dropout(p=0.1)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        # x = F.dropout(x,0.5)
        return self.classifier(x)

    def GetCriterion(self):
        return F.cross_entropy

    def GetOptimizer(self, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.0005)


def vgg_11():
    vgg_type = 'VGG11'
    Net = VGG(id='143756a6-4beb-11eb-804a-7085c27201ea').to(model.device)
    Net.name = lambda : 'VGG_11 without dropout'
    return Net

# def vgg_11():
#     vgg_type = 'VGG11'
#     # Net = VGG()

def vgg_13():
    vgg_type = 'VGG13'
    return VGG(id='12697fed-4bed-11eb-a172-7085c27201ea').to(model.device)


def vgg_16():
    vgg_type = 'VGG16'
    return VGG().to(model.device)


def vgg_19():
    vgg_type = 'VGG19'
    return VGG().to(model.device)


if __name__ == '__main__':
    print("Init The Net...")
    vgg_type = 'VGG11'
    net = vgg_11()
    print('网络初始化完成')
    net.train_and_test(epoch_count=100)
    print('训练完成')
