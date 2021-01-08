# Author: yuang
# @Time   : 2021/1/6 16:56
# @Author : yuang
# @Site   : www.bilegole.com
# @File   : YoloLoss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transform
import torchvision.datasets as datasets

import os, time, sys
from typing import TypeVar, Iterable, Tuple, List, Dict
import matplotlib.pyplot as plt

from pytorch.config import cache_path
from pytorch.common.utils import bbox_iou, bbox_wh_iou

device = "cuda:0" if torch.cuda.is_available() else "cpu"
Tensor = torch.Tensor


# class Output():
#     def __init__(self, Output: torch.Tensor, num_class):
#         # super(Output, self).__init__()
#         self.Value = Output
#         self.num_sample: int = self.Value.size(0)
#         self.num_anchor: int = self.Value.size(1)
#         self.num_grid: int = self.Value.size(2)
#
#     def desize(self):
#         return self.num_sample, self.num_anchor, self.num_grid
#
#     def dedata(self):
#         prediction = (self.Value.view(
#             self.num_sample, self.num_anchor, self.num
#         ))


class Targets(torch.Tensor):
    pass


class Anchor():
    '''
    anchor是指预选框，这里是单个的预选框，预选框的集合为
    '''
    pass


class YoloLoss():
    def __init__(self):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def check(self, output: Tensor):
        pass

    def process_outputs(self):
        pass

    def process_targets(self):
        pass

    def calculate(self):
        pass


Anchors = Tuple[Anchor]
ByteTensor = torch.cuda.ByteTensor if torch.cuda.is_available() else torch.ByteTensor
FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class YoloLoss_v1(YoloLoss):
    def check(self, output: Tensor):
        assert isinstance(output, Tensor)
        assert output.shape[1] == 7
        assert output.shape[2] == 7
        assert output.shape[3] == 30
        self.outputs = output

    def process_outputs(self):
        # px =
        pass


class YoloLoss_v3:
    def __init__(self, anchors: Anchors, classes: dict, sample_size: int, grid_size: int, img_dim: int = 416,
                 ignore_scale: float = 0.5, lambda_obj: int = 1, lambda_noobj=100):
        self.anchors: Anchors = anchors
        self.classes: Dict[int, str] = classes
        self.num_classes: int = self.classes.__len__()
        self.num_anchors = len(self.anchors)
        self.num_samples: int = sample_size
        self.img_dim = img_dim
        self.grid_size = grid_size
        self.ignore_scale = ignore_scale

        self.stride = self.img_dim / self.grid_size
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj

    def __call__(self, outputs: Tensor, targets: Targets):
        self.check(outputs)
        pred_x, pred_y, pred_w, pred_h, pred_conf, pred_cls = self.process_outputs()
        mask_obj, mask_noobj, tx, ty, tw, th, tcls, tconf = self.process_targets(target=targets)

        LossMSE = nn.MSELoss()
        loss_x = LossMSE(pred_x[mask_obj], tx[mask_obj])
        loss_y = LossMSE(pred_y[mask_obj], tx[mask_obj])
        loss_w = LossMSE(pred_w[mask_obj], tx[mask_obj])
        loss_h = LossMSE(pred_h[mask_obj], tx[mask_obj])
        loss_box = loss_x + loss_y + loss_w + loss_h

        LossBCE = nn.BCELoss()
        loss_conf_obj = LossBCE(pred_conf[mask_obj], tconf[mask_obj])
        loss_conf_noobj = LossBCE(pred_conf[mask_obj], tconf[mask_obj])
        loss_conf = self.lambda_obj * loss_conf_obj + self.lambda_noobj * loss_conf_noobj

        loss_cls = LossBCE(pred_cls[mask_obj], tcls[mask_obj])

        total_loss = loss_box + loss_conf + loss_cls
        return total_loss

    def check(self, output):
        '''
        主要是用于检查一下，输入进来的output尺寸对不对。
        Args:
            output:

        Returns:

        '''
        assert output.size(0) == self.num_samples
        assert output.size(1) == self.num_anchors
        assert output.size(2) == output.size(3) == self.grid_size
        assert output.size(4) == self.num_classes + 5
        self.output = output
        pass

    def process_outputs(self):
        '''
        本函数的作用是将
        Returns:

        '''
        self.prediction = (self.output.view(
            self.num_samples,
            self.num_anchors,
            self.num_classes + 4,
            self.grid_size, self.grid_size
        ).permute(-1, 1, 3, 4, 2).contiguous())

        pred_x = torch.sigmoid(self.prediction[..., 0])
        pred_y = torch.sigmoid(self.prediction[..., 1])
        pred_w = self.prediction[..., 2]
        pred_h = self.prediction[..., 3]
        pred_conf = self.prediction[..., 4]
        pred_cls = self.prediction[..., 5:]
        return [
            pred_x,
            pred_y,
            pred_w,
            pred_h,
            pred_conf,
            pred_cls
        ]

    def process_targets(self, target: Tensor):
        mask_shape = (self.num_samples, self.anchors, self.grid_size, self.grid_size)
        mask_obj = ByteTensor(*mask_shape).fill_(0)
        mask_noobj = ByteTensor(*mask_shape).fill_(1)
        tx = FloatTensor(*mask_shape).fill_(0)
        ty = FloatTensor(*mask_shape).fill_(0)
        tw = FloatTensor(*mask_shape).fill_(0)
        th = FloatTensor(*mask_shape).fill_(0)
        tcls = FloatTensor(*mask_shape).fill_(0)

        target_box = target[:, 2:6] * self.grid_size
        gxy, gwh = target_box[:, :2], target_box[:, 2:]
        ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in self.anchors])
        best_ious, best_ns = ious.max(0)
        image_index, target_labels = target[:, :2].long().t()
        gx, gy = gxy.t()
        gw, gh = gwh.t()
        gi, gj = gxy.long().t()

        # 将最有可能的网格设为对应预测
        mask_obj[image_index, best_ns, gi, gj] = 1
        mask_noobj[image_index, best_ns, gi, gj] = 0

        # 在mask_noobj中，将预期值满足要求的也置为0
        for i, anchor_ious in enumerate(ious.t()):
            mask_noobj[image_index[i], anchor_ious > self.ignore_scale, gj[i], gi[i]] = 0

        # 获取在网格内的相对网格坐标。
        tx[image_index, best_ns, gj, gi] = gx - gx.floor()
        ty[image_index, best_ns, gj, gi] = gy - gy.floor()
        tw[image_index, best_ns, gj, gi] = torch.log(gw / self.anchors[best_ns][:, 0] + 1e-16)
        th[image_index, best_ns, gj, gi] = torch.log(gh / self.anchors[best_ns][:, 1] + 1e-16)
        tcls[image_index, best_ns, gj, gi] = 1

        tconf = mask_obj.float()
        return mask_obj, mask_noobj, tx, ty, tw, th, tcls, tconf
