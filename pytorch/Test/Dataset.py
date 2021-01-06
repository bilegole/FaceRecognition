import unittest
from torchvision.datasets.vision import VisionDataset
from pytorch.config import cache_path
import torchvision.datasets as datasets
from PIL.Image import Image


class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, True)


class DatasetCase(unittest.TestCase):
    def general_dataset_test(self, TheClass: VisionDataset):
        pass

    def test_pytorch_mnist(self):
        trainset = datasets.MNIST(root=cache_path, train=True, download=True)
        testset = datasets.MNIST(root=cache_path, train=False, download=True)
        for dataset in [trainset, testset]:
            for index in range(0, len(dataset)):
                input, target = dataset.__getitem__(index)
                self.assertTrue(isinstance(input, Image), 'input的类型不是PIL.Image.Image')
                self.assertTrue(isinstance(target, int))
                self.assertTrue(target in range(0, 10))

    def test_CIFAR(self):
        trainset = datasets.CIFAR10(root=cache_path, train=True, download=True)
        testset = datasets.CIFAR10(root=cache_path, train=False, download=True)
        for dataset in [trainset, testset]:
            for index in range(0, len(dataset)):
                input, target = dataset.__getitem__(index)
                self.assertTrue(isinstance(input, Image))
                self.assertTrue(isinstance(target, int))

    def test_VOC(self):
        print('start')
        trainset = datasets.VOCDetection(root=cache_path, year='2012', image_set='train', download=False)
        trainvalset = datasets.VOCDetection(root=cache_path, year='2012', image_set='trainval', download=False)
        valset = datasets.VOCDetection(root=cache_path, year='2012', image_set='val', download=False)
        for dataset in [trainset, trainvalset, valset]:
            for index in range(0, len(dataset)):
                input, target = dataset.__getitem__(index)
                self.assertTrue(isinstance(target, dict))
                self.assertTrue('annotation' in target)
                self.assertTrue(len(target) == 1)
                self.assertTrue('object' in target['annotation'])
                for box in target['annotation']['object']:
                    for attr in ['name', 'pose', 'truncated', 'occluded', 'bndbox']:
                        self.assertTrue(attr in box)
                    tmp = box['bndbox']
                    self.assertEqual(4, len(tmp))
                    for attr in ['xmin', 'ymin', 'xmax', 'ymax']:
                        self.assertTrue(attr in tmp)
                        self.assertEqual(type(tmp[attr]), str)


if __name__ == '__main__':
    unittest.main()
