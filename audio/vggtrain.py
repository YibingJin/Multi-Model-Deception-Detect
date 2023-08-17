'''
基于PyTorch的VGG16迁移学习
'''

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from dataset import MyDataset
batch_size = 8
learning_rate = 0.0002
epoch = 10

train_path = './processedaudio/training set'
test_path = './processedaudio/test set'
train_datasets = MyDataset(train_path)
val_datasets = MyDataset(test_path)
model_path='./checkpoints'
"""
通过DataLoader将train_dataset载入进来
shuffle=True, 随机参数，随机从样本中获取数据
num_workers=0 加载数据所使用的线程个数，线程增多，加快数据的生成速度，也能提高训练的速度，
在Windows环境下线程数为0，在Linux系统下可以设置为多线程，一般设置为8或者16
"""
batch_size = 8
train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

val_num = len(val_datasets)
val_dataloader = torch.utils.data.DataLoader(val_datasets,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)




class VGGNet(nn.Module):
    def __init__(self, num_classes=2):
        super(VGGNet, self).__init__()
        net = models.vgg16(pretrained=True)  # 迁移学习，需要下载模型
        self.features = net
        self.classifier = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# --------------------训练过程---------------------------------
model = VGGNet()
if torch.cuda.is_available():
    device_name = "cuda:0"
else:
    device_name = "cpu"
device = torch.device(device_name)
model = model.to(device)
params = [{'params': md.parameters()} for md in model.children()
          if md in [model.classifier]]
optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=0.01)
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)
Loss_list = []
Accuracy_list = []

for epoch in range(30):
    print('epoch {}'.format(epoch + 1))
    # training-----------------------------
    train_loss = 0.
    train_acc = 0.
    for batch_x, batch_y in train_dataloader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

        out = model(batch_x)

        loss = loss_func(out, batch_y)

        train_loss += loss.item()
        pred = torch.max(out, 1)[1]

        train_correct = (pred == batch_y).sum()
        train_acc += train_correct.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Train Loss: {:.6f}, Acc: {:.6f}'.format(train_loss / (len(
        train_datasets)), train_acc / (len(train_datasets))))

    # evaluation--------------------------------
    model.eval()
    eval_loss = 0.
    eval_acc = 0.
    for batch_x, batch_y in val_dataloader:
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        loss = loss_func(out, batch_y)
        eval_loss += loss.item()
        pred = torch.max(out, 1)[1]
        num_correct = (pred == batch_y).sum()
        eval_acc += num_correct.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        val_datasets)), eval_acc / (len(val_datasets))))

    Loss_list.append(eval_loss / (len(val_datasets)))
    Accuracy_list.append(100 * eval_acc / (len(val_datasets)))
# 模型保存
torch.save(model, './model/model.pth')
# loss显示
x1 = range(0, 100)
x2 = range(0, 100)
y1 = Accuracy_list
y2 = Loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.show()
# plt.savefig("accuracy_loss.jpg")