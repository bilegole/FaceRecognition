import requests
import os

# author：yuang
# description：
#   主要目的：下载VGG数据集。
#   下载网站：https://www.robots.ox.ac.uk/~vgg/software/vgg_face/
#   下载地址：
downloads_links = {
    "torch.tar.gz": "https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz",
    "caffe.tar.gz": "https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz",
    "matconvnet.tar.gz": "https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_matconvnet.tar.gz"
}

#   下载到以下文件夹：
target = "../.cache/"
#

# 下载执行：


print(f"准备下载文件\n目标文件夹：{os.path.abspath(target)}\n")
for name in downloads_links:
    print(f"正在下载文件：{name}")
    remote_file = requests.get(downloads_links[name], stream=True)
    download_size = 0
    chuck_size = 512
    with open(os.path.join(target, name), 'wb') as file:
        for chunk in remote_file.iter_content(chunk_size=chuck_size):
            if chunk:
                file.write(chunk)
                download_size += 512
