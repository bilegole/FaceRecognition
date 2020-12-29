import pytorch.CIFAR.model as model


if __name__ == '__main__':
    net = model.DenseNet().to(model.device)
    net.train_and_test()
