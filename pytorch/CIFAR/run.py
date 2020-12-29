import pytorch.CIFAR.model as model


if __name__ == '__main__':
    print('准备中、、、')
    net = model.DenseNet().to(model.device)
    print("成功创建网络模型")
    net.train_and_test()
    print("网络训练完成")
