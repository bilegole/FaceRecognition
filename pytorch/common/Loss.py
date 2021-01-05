import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transform

import os
import sys

import pytorch.common.utils
from pytorch.common.utils import build_targets

Anchors = (
    (10, 13), (16, 30), (33, 23),
    (30, 61), (62, 45), (59, 119),
    (116, 90), (156, 198), (373, 326)
)


def YoloLoss(outputs, targets, num_classes:int=20, anchors=Anchors, img_dim=416,
             ignore_thres: float = 0.5):
    obj_scale = 1
    noobj_scale = 100

    # 指定了要使用的数据结构
    FloatTensor = torch.cuda.FloatTensor if outputs.is_cuda else torch.FloatTensor

    # 一些背景数据
    num_samples = outputs.size(0)  # 样本数量
    grid_size = outputs.size(2)  # S，即yolo将原始图片分割的小网格的边上的数量，一个网络最多输出S^2个对象。
    num_anchors = len(anchors)

    # reshape the outputs.
    # 反正就是将5种不同的输出(xywh和cls)移到最后一位，方便提取.
    # prediction 包含的是若干个线性排列的预测框。
    prediction = (
        outputs.view(num_samples, num_anchors, num_classes + 5, grid_size, grid_size)
            .permute(0, 1, 3, 4, 2).contiguous()
    )

    # 将输出中的不同部分分来开，方便后续分别计算loss.
    x = torch.sigmoid(prediction[..., 0])
    y = torch.sigmoid(prediction[..., 1])
    w = prediction[..., 2]
    h = prediction[..., 3]
    pred_conf = prediction[..., 4]
    pred_cls = torch.sigmoid(prediction[..., 5:])

    g = grid_size
    stride = img_dim / grid_size
    grid_x = torch.arange(g).repeat(g, 1).view([1, 1, g, g]).type(FloatTensor)
    grid_y = torch.arange(g).repeat(g, 1).t().view([1, 1, g, g]).type(FloatTensor)

    # 给出的物体框
    scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in anchors])
    anchor_w = scaled_anchors[:, 0:1].view((1, num_anchors, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, num_anchors, 1, 1))

    pred_boxes = FloatTensor(prediction[..., 4].shape)
    pred_boxes[..., 0] = x.data + grid_x
    pred_boxes[..., 1] = y.data + grid_y
    pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
    pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

    output = torch.cat((
        pred_boxes.view(num_samples, -1, 4) * stride,
        pred_conf.view(num_samples, -1, 1),
        pred_cls.view(num_samples, 1, num_classes),
    ), 1)

    iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf = build_targets(
        pred_boxes=pred_boxes,
        pred_cls=pred_cls,
        target=targets,
        anchors=scaled_anchors,
        ignore_thres=ignore_thres
    )

    LossMSE = nn.MSELoss()
    loss_x = LossMSE(x[obj_mask], tx[obj_mask])
    loss_y = LossMSE(y[obj_mask], tx[obj_mask])
    loss_w = LossMSE(w[obj_mask], tw[obj_mask])
    loss_h = LossMSE(h[obj_mask], th[obj_mask])
    loss_box = loss_x + loss_y + loss_w + loss_h

    LossBCE = nn.BCELoss()
    loss_conf_obj = LossBCE(pred_conf[obj_mask], tconf[obj_mask])
    loss_conf_noobj = LossBCE(pred_conf[noobj_mask], tconf[noobj_mask])
    loss_conf = obj_scale * loss_conf_obj + noobj_scale * loss_conf_noobj
    loss_cls = LossBCE(pred_cls[obj_mask], tcls[obj_mask])

    total_loss = loss_box + loss_conf + loss_cls
