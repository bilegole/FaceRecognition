import torch.nn as nn
from history.VOC import GenModel
from history.VOC import sconv, mslayer, savgpool, sConnected, sSoftmax

from collections import OrderedDict

para = ['batch_normalize', 'filter', 'size', 'stride', 'pad', 'activation']
YoloModel = [
    sconv(32, 3, 1, 1, 'conv_1'),
    sconv(64, 3, 2, 1, 'conv_2'),
    mslayer(1, [
        sconv(32, 1, 1, 1, 'm1_conv1'),
        sconv(64, 3, 1, 1, 'm1_conv2'),
    ]),
    sconv(128, 3, 2, 1, 'conv_3'),
    mslayer(2, [
        sconv(64, 3, 2, 1, 'm2_conv1'),
        sconv(128, 3, 1, 1, 'm2_conv1'),
    ]),
    sconv(256, 3, 2, 1, 'conv_4'),
    mslayer(8, [
        sconv(128, 3, 2, 1, 'm3_conv1'),
        sconv(256, 3, 1, 1, 'm3_conv1'),
    ]),
    sconv(512, 3, 2, 1, 'conv_5'),
    mslayer(8, [
        sconv(256, 1, 1, 1, 'm4_conv1'),
        sconv(512, 3, 1, 1, 'm4_conv2'),
    ]),
    sconv(1024, 3, 2, 1, 'conv_6'),
    mslayer(4, [
        sconv(512, 1, 1, 1, 'm5_conv1'),
        sconv(1024, 3, 1, 1, 'm5_conv2'),
    ]),
    savgpool(),
    sConnected(),
    sSoftmax(),
]




class Darknet_Main(GenModel):
    def __init__(self, config_path):
        super(Darknet_Main, self).__init__()
        # self.module_def = parse_model_config(config_path)

    def config_layers(self):
        return {
            'conv_1': {

            }
        }
