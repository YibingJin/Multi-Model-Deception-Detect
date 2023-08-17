import os
import random
import torch
from torch.utils import data
import numpy as np
import pandas as pd


#
class Mydataset(data.Dataset):
    def __init__(self, txt_name, train=True, transform=None, target_tranform=None, loader=None, identify=None):
        super(Mydataset, self).__init__()
        # self.img_label =
        line_list = []
        if train:
            file_name = r'./trainfold2.txt'

        else:
            file_name = r'./testfold2.txt'

        with open(file_name, 'r') as fp:
            lines = fp.readlines()
            for i in lines:
                line = i.split()
                line_list.append(line)



        self.img = line_list
        self.transform = transform

        self.target_transform = target_tranform
        self.loader = loader

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        path, label = self.img[index]
        label = np.array(float(label))

        img = pd.read_csv(path,header=None)  # 读取出来维度为3， 22， 14 ？？ 无法正常显示图片


        # img = Image.open(path)
        img = np.array(img, dtype=np.float64)
        # # img = np.array(img, dtype=np.float32).reshape(14, 22, 3)
        img = torch.from_numpy(img)
        video_length, fea_dim = img.shape
        pad_length = 30-video_length%30
        pad_zero = torch.zeros((pad_length, fea_dim))
        result = torch.cat((img, pad_zero), dim=0)
        result = result.reshape(-1, 30, 43).float()
        Length = result.shape[0]
        return result, label, Length,index

if __name__ == "__main__":
    # mydata = Mydataset('name.txt')
    # print(mydata.__getitem__(5))

    train_data = Mydataset(
        txt_name='name.txt',  # 暂时未使用
        train=True,
    )
    # data_train = DataLoader(train_data, batch_size=64)
    img, label, Length,index= train_data[20]
    a=1
