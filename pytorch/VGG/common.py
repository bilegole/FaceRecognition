import torchfile
import os
# data_path = '.cache/vgg_face_torch/VGG_FACE.t7'
# os.path.exists(data_path)
# # data = torchfile.load(data_path)

from urllib.parse import urlparse
from urllib.request import Request,ProxyHandler
import torch.utils.model_zoo as model_zoo
import re
import os

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}
path = os.path.join('.cache','vgg')

def download_model(url, dst_path):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)

    HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
    hash_prefix = HASH_REGEX.search(filename).group(1)
    model_zoo.load_url(url, os.path.join(dst_path, filename))
    return filename

def downloads():
    if not (os.path.exists(path)):
        os.makedirs(path)
    for url in model_urls.values():
        download_model(url, path)

if __name__ == '__main__':
    downloads()
