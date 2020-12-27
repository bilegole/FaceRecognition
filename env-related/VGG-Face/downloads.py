# author：yuang
# description：
#   主要目的：下载VGG数据集。
#   下载网站：https://www.robots.ox.ac.uk/~vgg/software/vgg_face/
#   下载地址：
downloads_links{
    "torch":"https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz",
    "caffe":"https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz",
    "matconvnet":"https://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_matconvnet.tar.gz"
}
#   下载到以下文件夹：
target="../.cache/"
# 

# 下载执行：
import requests
print