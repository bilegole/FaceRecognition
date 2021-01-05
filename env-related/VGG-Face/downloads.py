import sys
import time

# import requests
import urllib.request as request
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
target = "../../.cache/"
#   可用代理：
proxy={"HTTP":"192.168.10.13:10809"}



'''
 urllib.urlretrieve 的回调函数：
def callbackfunc(blocknum, blocksize, totalsize):
    @blocknum:  已经下载的数据块
    @blocksize: 数据块的大小
    @totalsize: 远程文件的大小
'''
def Schedule(blocknum, blocksize, totalsize):
    speed = (blocknum * blocksize) / (time.time() - start_time)
    # speed_str = " Speed: %.2f" % speed
    speed_str = " Speed: %s" % format_size(speed)
    recv_size = blocknum * blocksize

    # 设置下载进度条
    f = sys.stdout
    pervent = recv_size / totalsize
    percent_str = "%.2f%%" % (pervent * 100)
    n = round(pervent * 50)
    s = ('#' * n).ljust(50, '-')
    f.write(percent_str.ljust(8, ' ') + '[' + s + ']' + speed_str)
    f.flush()
    # time.sleep(0.1)
    f.write('\r')


# 字节bytes转化K\M\G
def format_size(bytes):
    try:
        bytes = float(bytes)
        kb = bytes / 1024
    except:
        print("传入的字节格式不对")
        return "Error"
    if kb >= 1024:
        M = kb / 1024
        if M >= 1024:
            G = M / 1024
            return "%.3fG" % (G)
        else:
            return "%.3fM" % (M)
    else:
        return "%.3fK" % (kb)


print(f"准备下载文件\n目标文件夹：{os.path.abspath(target)}\n")
for name in downloads_links:
    print(f"正在下载文件：{name},输出到:\n\t{os.path.abspath(target)}\n")
    start_time = time.time()
    request.ProxyHandler(proxy)
    request.urlretrieve(
        url=downloads_links[name],
        filename=os.path.join(target,name),
        reporthook=Schedule
    )



