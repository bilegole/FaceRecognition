import history.CIFAR.model as model
import torch
import torch.nn as nn


# device = "cuda:0" if torch.cuda.is_available() else "cpu"


class DenseNet_1(model.GeneralNetwork):
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
    net = DenseNet_1(id="1150c22c-4b22-11eb-99eb-a4b1c131cc51").to(model.device)
    print("网络初始化完成，开始训练")
    net.train_and_test(epoch_count=20)
    print("网络训练完成")
