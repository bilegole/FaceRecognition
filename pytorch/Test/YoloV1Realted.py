import time
import unittest
import torch
import torchvision.models as models

import pytorch.Loss.YoloLoss as yll
import pytorch.Models.Yolo as Models
import pytorch.Practice.Yolo_V1 as P

FloatTensor = torch.FloatTensor
Tensor = torch.Tensor

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


# model = models

class WithModelCase(unittest.TestCase):
    def test_dry_step(self):
        start_time = time.time()
        model: P.YoloV1 = P.YoloV1().to(device)
        model_inited = time.time()
        print(f'初始化模型耗时{model_inited-start_time:3f}秒')
        self.assertEqual(type(model), P.YoloV1)
        self.assertTrue(hasattr(model, 'GetTrainLoader'), 'model 没有GetTrainLoader函数，疑似继承不正确.')
        # loader = model.GetTrainLoader()
        dataset = model.GetTrainDataset()
        dataset_generated = time.time()
        print(f"数据库初始化耗时{dataset_generated-model_inited}秒")
        batch = [dataset.__getitem__(i) for i in range(10)]
        processed_batch = model.GetTrainCollectFun(batch)
        # for inputs, targets in loader:
        inputs, targets = processed_batch
        inputs: Tensor
        targets: Tensor
        inputs, targets = inputs.to(device), targets.to(device)
        data_prepared = time.time()
        print(f"生成一个batch耗时{data_prepared-dataset_generated}秒")
        # ----------------------------
        self.assertEqual(Tensor, type(inputs))
        self.assertEqual(Tensor, type(targets))
        self.assertEqual(10, inputs.shape[0])
        # ----------------------------
        outputs: Tensor = model(inputs)
        output_cal = time.time()
        print(f"正向传播耗时{output_cal-data_prepared}秒")
        self.assertEqual(Tensor, type(outputs))
        self.assertEqual(10, outputs.shape[0])
        self.assertEqual(7, outputs.shape[1])
        self.assertEqual(7, outputs.shape[2])
        self.assertEqual(30, outputs.shape[3])

        # ----------------------------
        loss = model.criterion(outputs, targets)
        loss_cal = time.time()
        print(f'计算loss，耗时{loss_cal-output_cal}秒')
        loss.backward()
        backward = time.time()
        print(f'反向传播，耗时{backward-loss_cal}秒')
        print(loss)

        optim, sche = model.GetOptimizer()
        optim.step()
        optim_cal = time.time()
        print(f"优化模型，一步耗时{optim_cal-backward}秒")
        # ----------------------------

    def test_train(self):
        model = P.YoloV1().to(device)
        model.train_and_test(10)


class V1LossCase(unittest.TestCase):
    def test_init_the_loss_fun(self):
        yll.YoloLoss_v1(num_sample=10)

    def test_process_output_static(self):
        LossFun = yll.YoloLoss_v1(num_sample=10)
        output = FloatTensor(10, 7, 7, 30).fill_(0)
        target = FloatTensor(20, 6).fill_(0)
        loss = LossFun(output, target)


if __name__ == '__main__':
    unittest.main()
