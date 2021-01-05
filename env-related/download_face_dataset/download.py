import os
import numpy as np
import urllib.request as request
import urllib
# from urllib.request import ProxyHandler,
import hashlib
from multiprocessing import Process
from multiprocessing import  pool as Pool
def download_photo(name,index,url,rect,md5):
    if not os.path.exists(os.path.join(data_path,name,index+'.jpg')):
        print(f"下载{name}的第{index}张照片.")
        try:
            for i in range(3):
                r = urllib.request.urlopen(url)
                data = r.read()
                m = hashlib.md5()
                m.update(data)
                if md5 != m.hexdigest():
                    continue
                dir_path = os.path.join(data_path,name)
                if not os.path.exists(dir_path):
                    os.mkdir(dir_path)
                with open(os.path.join(dir_path,index+'.jpg'),'wb') as photo:
                    photo.write(data)
                break
            print(f"成功下载{name}的第{index}张照片")
        except:
            print(f"{name}的第{index}张图片下载失败.")

if True:
    handler = request.ProxyHandler({"HTTP":"127.0.0.1:10809"})
    opener = request.build_opener(handler)
    request.install_opener(opener)



filename = 'dev_urls.txt'
cache_path = os.path.abspath('../../.cache')
data_path = os.path.join(cache_path,'PubFig')
if not os.path.exists(data_path):
    os.mkdir(data_path)
with open(filename,'r') as f:
    files = f.readlines()
column_name = files[1].strip('\n').split('\t')[1:]
print(column_name)
name_list = set([line.split('\t')[0] for line in files[2:]])
print(name_list)
print(len(name_list))

if __name__ == '__main__':
    pool = Pool.Pool(50)
    for index,line in enumerate(files[2:]):
        print(f"第{index}个进程创建")
        state = line.strip('\n').split('\t')
        pool.apply_async(download_photo,args=state)
    # print(state)
    # try:
    #     # r = urllib.request.Request(url=state[2])
    #     r = urllib.request.urlopen(url=state[2])
    #     data = r.read()
    #     # md5 = hashlib.md5()
    #     # md5.update(data)
    #     # md5_str = md5.hexdigest()
    #     # # if md5_str==state[-1]:
    #     dir_path = os.path.join(data_path,state[0])
    #     # os.mkdir(os.path.join(data_path,state))
    #     if not os.path.exists(dir_path):
    #         os.mkdir(dir_path)
    #     with open(os.path.join(dir_path,state[1]+'.jpg'),'wb') as photo:
    #         photo.write(data)
    # except:
    #     print('Error in {}'.format(index))
    #     pass


