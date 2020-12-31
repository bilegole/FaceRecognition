import pytorch.CIFAR.model as model
import torch
import torch.nn as nn
import torch.nn.functional as fun

# device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DenseNet_1(model.GeneralNetwork):
    def __init__(self,id=None):
        super(DenseNet_1, self).__init__(id=id)

    def init_layers(self):
        # self.input_size = 32 * 32 * 3
        # self.hidden_layer = nn.Linear(self.input_size, 100)
        self.output_layer = nn.Linear(32*32*3, 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 32*32*3)
        # x = self.hidden_layer(x)
        return self.output_layer(x)


if __name__ == '__main__':
    print('开始初始化网络。。。')
    # net = model.DenseNet_1(id="922afef5-4b0a-11eb-bff5-7185c27201ea").to(model.device)
    net = DenseNet_1().to(model.device)
    print("网络初始化完成，开始训练")
    net.train_and_test(epoch_count=20)
    print("网络训练完成")
