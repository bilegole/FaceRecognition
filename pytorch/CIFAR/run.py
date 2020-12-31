import pytorch.CIFAR.model as model

def DenseNet():
    return model.DenseNet(id="7b86d287-4b0a-11eb-9a85-7085c27201ea").to(model.device)

def DenseNet_1():
    return model.DenseNet_1(id="922afef5-4b0a-11eb-bff5-7085c27201ea").to(model.device)

if __name__ == '__main__':
    print('准备中、、、')
    # net = model.DenseNet(id=None).to(model.device)
    net= DenseNet_1()
    print("成功创建网络模型")
    net.train_and_test(20)
    print("网络训练完成")
