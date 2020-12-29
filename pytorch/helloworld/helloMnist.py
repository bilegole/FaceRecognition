import torch
import torch.nn as nn
from pytorch.helloworld.common import (get_train_loader, IMAGE_WIDTH, train_network, get_test_loader, test_network)

INPUT_SIZE = IMAGE_WIDTH * IMAGE_WIDTH
OUTPUT_SIZE = 10
NUM_EPOCHS = 3
LEARNING_RATE = 3.0


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden_layer = nn.Linear(INPUT_SIZE, 100)
        self.output_layer = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = x.reshape(-1, INPUT_SIZE).to(device)
        x = torch.sigmoid(self.hidden_layer(x)).to(device)
        x = torch.sigmoid(self.output_layer(x)).to(device)
        return x


def expand_expected_output(tensor_of_expected_outputs, output_size):
    return torch.tensor([expand_single_output(expected_output.item(),
                                              output_size)
                         for expected_output in tensor_of_expected_outputs])


# Expand expected output for comparison with the outputs from the network,
# e.g. convert 3 to [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
def expand_single_output(expected_output, output_size):
    x = [0.0 for _ in range(output_size)]
    x[expected_output] = 1.0
    return x


def create_loss_function(loss_function, output_size=OUTPUT_SIZE):
    def calc_loss(outputs, target):
        target = expand_expected_output(target, output_size).to(device)
        return loss_function(outputs, target)

    return calc_loss


def run_network(net):
    train_loader = get_train_loader()
    mse_loss_function = nn.MSELoss().to(device)
    loss_function = create_loss_function(mse_loss_function)
    sgd = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    train_network(net, train_loader, NUM_EPOCHS, loss_function, sgd)
    print("")

    test_loader = get_test_loader()
    test_network(net, test_loader)

# def Gpu(target):
#     if gpu:
#         return target.cuda()
#     else:
#         return target

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using {device}")
    print("start ")
    network = Net().to(device)
    run_network(network)
