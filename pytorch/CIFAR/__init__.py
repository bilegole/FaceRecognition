import pandas as pd
import pickle
import os

CACHE_PATH = os.path.abspath('../../.cache')
data = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5'
]
# file = os.path.join(CACHE_PATH, 'CIFAR', 'data_batch_1')
# with open(file, 'rb') as f:
#     dic = pickle.load(f, encoding='bytes')
#
# print(dic)


def loader(isTrain=True):
    pass


def train_network():
    pass


def test_network():
    pass
