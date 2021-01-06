import collections
import os
import sys
import time
import xml.etree.ElementTree as ET

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

# from torchvision.transforms.transforms import
import torchvision.transforms as transform
from PIL import Image
from torchvision.datasets.vision import VisionDataset

from history.VOC.Utils import GenerateID
# from pytorch.VOC import progress_bar
from history.VOC.Utils import progress_bar

path = os.path.abspath('../../../.cache')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
total_len = 10


class VOCDataset(VisionDataset):
    def __init__(
            self, root: str = path,
            year: str = '2028',
            train:bool= True,
            download: bool = False,
            transform=None,
            target_transform=None,
            transforms=None
    ):
        super(VOCDataset, self).__init__(
            root=root, transform=transforms, transforms=transform, target_transform=target_transform
        )
        base_dir = root
        voc_dir = os.path.join(base_dir, 'VOC')
        image_dir = os.path.join(voc_dir, 'JPEGImages')
        annotation_dir = os.path.join(voc_dir, 'Annotations')

        image_set='train' if train else 'test'

        splits_dir = os.path.join(voc_dir, 'ImageSets', 'Main')
        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        with open(split_f, 'r') as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + '.jpg') for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + '.xml') for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index: int):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot()
        )
        if self.transform:
            img, target = self.transforms(img, target)
        return img, target

    def parse_voc_xml(self, node: ET.Element):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def __len__(self):
        return len(self.images)


# DataLoader = transform.Compose([
#     transform.ToTensor(),
#     transform.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])


class GenModel(nn.Module):
    def __init__(self, cache_path=path, dir_name='VOC', id=None, verbose: bool = False):
        super(GenModel, self).__init__()
        self.root_path = cache_path
        self.checkpoint = os.path.join(cache_path, dir_name, 'checkpoint')
        self.verbose = verbose

        try:
            if self.verbose:
                print("开始模型初始化工作,自检中...")
            self._init_layers()
            self.id = GenerateID(id)
            self.optimizer = self.GetOptimizer()
            self.train_loader = self.GetTrainLoader()
            self.test_loader = self.GetTestLoader()
            self.criterion = self.GetCriterion()
        except:
            print("模型初始化失败.")
        else:
            print("\t模型初始化完成.")

        self.acc: float = 0.
        self.best_acc: float = 0.
        self.epoch: int = 0
        self.history = dict()
        # self.name
        try:
            if self.verbose:
                print(f"尝试从{os.path.join(self.checkpoint, self.id + '.pth')}导入存档")
            self.load_checkpoint()
        except:
            raise Exception('导入模型失败.')

    def _init_layers(self):
        pass

    def name(self):
        return self._get_name()

    def description(self):
        return 'unknown'

    def GetTrainTransform(self):
        pass

    def GetTrainLoader(self):
        print(f'请重写GetTrainLoader')
        # sys.exit(0)

    def GetTestTransform(self):
        return transform.Compose([])

    def GetTestLoader(self):
        pass

    def GetOptimizer(self):
        try:
            optimizer = optim.SGD(self.parameters(), lr=0.005)
        except:
            print(f"优化器初始化出错,请检查模型{self.name()}")
            # sys.exit(0)
            optimizer = None
        return optimizer

    def GetCriterion(self):
        print(f"{self.name()}未完成函数\"{sys._getframe().f_code.co_name}\"")
        pass

    def the_back_up(self):
        return os.path.join(self.checkpoint, self.id.__str__() + '.pth')

    def save_checkpoint(self) -> None:
        checkpoint = {
            'net': self.state_dict(),
            'acc': self.acc,
            'epoch': self.epoch,
            'id': self.id,
            'name': self.name(),
            'description': self.description(),
            'history': self.history
        }
        if not os.path.exists(self.checkpoint):
            os.mkdir(self.checkpoint)
        torch.save(checkpoint, self.the_back_up())

    def load_checkpoint(self) -> None:
        if os.path.exists(self.the_back_up()):
            checkpoint = torch.load(self.the_back_up())
            try:
                self.load_state_dict(checkpoint['net'])
                self.best_acc = checkpoint['acc']
                self.epoch = checkpoint['history']
                self.history = checkpoint['history']
            except:
                if input("网络结构已发生改变，是否覆盖存档？(yes/no)") == 'yes':
                    sys.exit(0)
        else:
            print('未发现存档')

    def train_and_test(self, epoch_count: int = 1) -> None:
        tmp = self.epoch + 1
        for epoch in range(tmp, tmp + epoch_count):
            self.epoch = epoch
            self.Train(epoch, tmp + epoch_count)
            self.Test()

        pass

    def Train(self, epoch: int, max_epoch: int) -> None:
        self.train()
        train_loss = 0.
        correct = 0
        total = 0
        start_time = time.time()
        for batch_index, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            self.optimizer.zero_grad()
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            progress_bar((epoch, max_epoch), (batch_index, len(self.train_loader)), bar_len=10,
                         start_time=int(start_time), loss=loss.item(), acc=correct / total)
            # print(f"\b{epoch:3d}/{max_epoch:3d}|[{batch_index/}]")
        pass

    def Test(self) -> None:
        self.eval()
        test_loss = 0.
        correct = 0
        total = 0
        start = time.time()
        with torch.no_grad():
            for batch_index,(inputs,targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(device),targets.to(device)
                outputs = self(targets)
                loss = self.criterion(outputs,targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        print(f"The test | Time:{time.time()-start:2.2f} | Loss: {test_loss/len(self.test_loader):4.2f} |"
                     f" Acc:{correct/total:3.2f}%")

    def forward(self, x):
        return x

class VOC_Model(GenModel):
    def GetTrainTransform(self):
        return transform.Compose([
            transform.RandomCrop(32,padding=4),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
        ])

    def GetTestTransform(self):
        return transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])

    def GetTrainLoader(self):
        return DataLoader(VOCDataset(),batch_size=100,shuffle=True,num_workers=2)

    def GetTestLoader(self):
        return DataLoader(VOCDataset(),batch_size=1000,shuffle=True,num_workers=2)


if __name__ == '__main__':
    # # a = VOCDataset(root=path)
    # # img,target = a.__getitem__(0)
    # # import matplotlib.pyplot as plt
    # # plt.imshow(img)
    # # plt.show()
    # a = GenModel(cache_path=path, dir_name='VOC')
    # # print(a._get_name())
    # progress_bar((10, 20), (34, 100), 10, 1000, 3.222, 34.123)
    # print('\n---\n')
    # progress_bar((10, 20), (34, 100), 10, 1000, 3.222, 34.123)
    # progress_bar((10, 20), (56, 100), 10, 1000, 2.102, 54.223)
    from torchvision.datasets import celeba
    celeba.CelebA(root=path,download=True)
