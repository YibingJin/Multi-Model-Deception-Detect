import os
import ssl
import time

import torch
import torchvision
import torchvision.transforms as transforms
import math
import torch.nn as nn
import copy

from dataset import MyDataset
# coding=UTF-8
import torchvision.models as models


train_path = './processedaudio/training set'
test_path = './processedaudio/test set'
train_dataset = MyDataset(train_path)
test_dataset = MyDataset(test_path)
model_path='./checkpoints'
"""
通过DataLoader将train_dataset载入进来
shuffle=True, 随机参数，随机从样本中获取数据
num_workers=0 加载数据所使用的线程个数，线程增多，加快数据的生成速度，也能提高训练的速度，
在Windows环境下线程数为0，在Linux系统下可以设置为多线程，一般设置为8或者16
"""
batch_size = 8
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

val_num = len(test_dataset)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)



# 设置网络参数的trainable属性, 即设置梯度迭代使能的属性
def set_model_grad_state(model, trainable_state):
    for param in model.parameters():
        param.requires_grad = trainable_state

# 显示网络参数允许trainable的参数,即梯度迭代使能的参数
def show_model_grad_state_enabled(model):
    print("params to be trained:")
    for name, parameters in model.named_parameters():
        if(parameters.requires_grad == True):
            print(name, ':', parameters.requires_grad)


# model_name: 模型的名称
# num_classes：输出种类
# lock_feature_extract：是否锁定特征提取网络
# use_pretrained：是否需要使用预训练参数初始化自定义的神经网络
# feature_extact_trainable: 特征提取层是否能够训练，即是否需要锁定特征提取层
def initialize_model(model_name, num_classes, use_pretrained=False, feature_extact_trainable=True):
    model = None
    input_size = 0

    if (model_name == "resnet"):
        if (use_pretrained == True):
            # 使用预训练参数
            model = models.resnet101(pretrained=True)

            # 锁定特征提取层
            set_model_grad_state(model, feature_extact_trainable)

            # 替换全连接层
            num_in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(num_in_features, num_classes))
        else:
            model = models.resnet101(pretrained=False, num_classes=num_classes)
        input_size = 224

    elif (model_name == "vgg"):
        if (use_pretrained == True):
            # 使用预训练参数
            model = models.vgg16(pretrained=True)

            # 锁定特征提取层
            set_model_grad_state(model, feature_extact_trainable)

            # 替换全连接层
            num_in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(nn.Linear(num_in_features, num_classes))
        else:
            model = models.vgg16(pretrained=False, num_classes=num_classes)
        input_size = 224
    return model, input_size


# 创建网络实例
model, input_size = initialize_model(model_name="vgg", num_classes=2, use_pretrained=True,
                                     feature_extact_trainable=False)

print(input_size)
print(model)


# 模块迁移学习/训练的定义：
# 一边在训练集上训练，一边在验证集上验证
# 策略：
# 最终选择在整个验证集上，而不是验证集的一个batch上，其准确率最高的模型参数以及优化器参数作为最终的模型参数
# 在整个验证集，而不是batch的目的：增加在测试集上的泛化能力
# 在验证集上准确率最高的目的：     防止在训练集上的过拟合
def model_train(model, train_loader, test_loader, criterion, optimizer, device, num_epoches, check_point_filename=""):
    # 记录训练的开始时间
    time_train_start = time.time()
    print('+ Train start: num_epoches = {}'.format(num_epoches))

    # 历史数据，用于显示
    batch_loss_history = []
    batch_accuracy_history = []
    best_accuracy_history = []

    # 记录最好的精度，用于保存此时的模型，并不是按照epoch来保存模型，也不是保存最后的模型
    best_accuracy = 0
    best_epoch = 0

    # 使用当前的模型参数，作为best model的初始值
    best_model_state = copy.deepcopy(model.state_dict())

    # 把模型迁移到 GPU device上
    model.to(device)

    # epoch层
    for epoch in range(num_epoches):
        time_epoch_start = time.time()
        print('++ Epoch start: {}/{}'.format(epoch, num_epoches - 1))

        epoch_size = 0
        epoch_loss_sum = 0
        epoch_corrects = 0
        highest_acc = 0

        # 数据集层
        # 每训练完一个epoch，进行一次全训练样本的训练和一次验证样本的验证
        for dataset in ["train", "valid"]:
            time_dataset_start = time.time()
            print('+++ dataset start: epoch = {}, dataset = {}'.format(epoch, dataset))

            if dataset == "train":
                model.train()  # 设置在训练模式
                data_loader = train_loader
            else:
                model.eval()  # 设置在验证模式
                data_loader = test_loader

            dataset_size = len(data_loader.dataset)
            dataset_loss_sum = 0
            dataset_corrects = 0

            # batch层
            # begin to operate in mode
            for batch, (inputs, labels) in enumerate(data_loader):
                # (0) batch size
                batch_size = inputs.size(0)


                # (1) 指定数据处理的硬件单元
                inputs = inputs.to(device)
                labels = labels.to(device)

                # (2) 复位优化器的梯度
                optimizer.zero_grad()

                # session层
                with torch.set_grad_enabled(dataset == "train"):
                    # (3) 前向计算输出
                    outputs = model(inputs)

                    # (4) 计算损失值
                    loss = criterion(outputs, labels)

                    if (dataset == "train"):
                        # (5) 反向求导
                        loss.backward()

                        # (6) 反向迭代
                        optimizer.step()

                    # (7-1) 统计当前batch的loss(包括训练集和验证集)
                    batch_loss = loss.item()

                    # (7-2) # 统计当前batch的正确样本的个数和精度(包括训练集和验证集)
                    # 选择概率最大的索引作为分类值
                    _, predicteds = torch.max(outputs, 1)
                    batch_corrects = (predicteds == labels.data).sum().item()
                    batch_accuracy = 100 * batch_corrects / batch_size
                    if (dataset == "valid"):
                        print('predicted:',predicteds)
                        print('labels:',labels.data)

                    # （8-1）统计当前dataset总的loss(包括训练集和验证集)
                    dataset_loss_sum += batch_loss * batch_size

                    # （8-2）统计当前dataset正确样本的总数(包括训练集和验证集)
                    dataset_corrects += batch_corrects

                    # 把训练结果添加到history log，用于后期的图形显示
                    batch_loss_history.append(batch_loss)
                    batch_accuracy_history.append(batch_accuracy)

                if (batch % 8 == 0):
                    print(
                        '++++ batch done: epoch = {}, dataset = {}, batch = {}/{}, loss = {:.4f}, accuracy = {:.4f}%'.format(
                            epoch, dataset, batch, dataset_size // batch_size, batch_loss, batch_accuracy))

                    # 统计dataset的平均loss
            dataset_loss_average = dataset_loss_sum / dataset_size

            # 统计dataset的平均准确率
            dataset_accuracy_average = 100 * dataset_corrects / dataset_size

            # 统计当前epoch总的loss
            epoch_loss_sum += dataset_loss_sum

            # 统计当前epoch总的正确数
            epoch_corrects += dataset_corrects

            # epoch_size
            epoch_size += dataset_size

            # 模型保存：此处策略为：在验证集上，每次精度提升的时候，都保存一次模型参数，防止过拟合
            if (dataset == "valid") and (dataset_accuracy_average > best_accuracy):
                # 保存当前的最佳精度(防止过拟合)
                best_accuracy = dataset_accuracy_average
                # 保存最佳epoch（检查是否有过拟合训练）
                best_epoch = epoch

                print('+++ model save with new best_accuracy = '.format(best_accuracy))
                # 获取当前的模型参数
                best_model_state = copy.deepcopy(model.state_dict())
                state = {
                    "state_dict": model.state_dict(),
                    "best_accuracy": best_accuracy,
                    "optimizer": optimizer.state_dict(),
                }
                if (check_point_filename != ""):
                    torch.save(state, check_point_filename)

                best_accuracy_history.append(best_accuracy)

            time_dataset_done = time.time()
            time_dataset_elapsed = time_dataset_done - time_dataset_start
            print(
                '+++ dataset done：epoch = {}, dataset = {}, loss = {:.4f}, accuracy = {:.4f}%, elapsed time = {:0f}m {:.0f}s'.format(
                    epoch, dataset, dataset_loss_average, dataset_accuracy_average, time_dataset_elapsed // 60,
                    time_dataset_elapsed % 60))

            # 统计epoch的平均loss
        epoch_loss_average = epoch_loss_sum / epoch_size

        # 统计epoch的平均正确率
        epoch_accuarcy_average = 100 * epoch_corrects / epoch_size

        time_epoch_done = time.time()
        time_epoch_elapsed = time_epoch_done - time_epoch_start

        print(
            '++ epoch done: epoch = {}, loss = {:.4f}, accuracy = {:.4f}%, elapsed time = {:0f}m {:.0f}s'.format(epoch,
                                                                                                                 epoch_loss_average,
                                                                                                                 epoch_accuarcy_average,
                                                                                                                 time_epoch_elapsed // 60,
                                                                                                                 time_epoch_elapsed % 60))

        # 恢复最佳模型
    model.load_state_dict(best_model_state)

    # 记录训练的结束时间
    time_train_done = time.time()
    time_train_elapsed = time_train_done - time_train_start
    print('+ Train Finished: elapsed time = {:0f}m {:.0f}s'.format(time_train_elapsed // 60, time_train_elapsed % 60))
    print('+++ model save with new best_accuracy = ',max(best_accuracy_history))
    print('_________________________________________________________________________________')

    return (model, batch_loss_history, batch_accuracy_history, best_accuracy_history)

# 指定loss函数
loss_fn = nn.CrossEntropyLoss()
#loss_fn = nn.NLLLoss()
# 指定优化器
Learning_rate = 0.01  # 学习率

# optimizer = SGD： 基本梯度下降法
# parameters：指明要优化的参数列表
# lr：指明学习率
optimizer = torch.optim.Adam(model.parameters(), lr = Learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=Learning_rate, momentum=0.9)
print(optimizer)

# 训练前准备
# 检查是否支持GPU，如果支持，则使用GPU训练，否则，使用CPU训练
if torch.cuda.is_available():
    device_name = "cuda:0"
else:
    device_name = "cpu"

# 生成torch的device对象
device = torch.device(device_name)
print(device)

# 把模型计算部署在GPUS上
model = model.to(device)

# 把loss计算转移到GPU
loss_fn = loss_fn.to(device)  # 自适应选择法
# loss_fn.cuda()                 # 强制指定法



# 保存训练后的模型
model_trained_path = "./models/checkpoint.pth"

# 定义迭代次数
epochs = 30
checkpoint_file = "./checkpoints/vgg16net_checkpoint.pth"
model, batch_loss_history, batch_accuracy_history, best_accuracy_history = model_train(
                                                                                model = model,
                                                                                train_loader = train_loader,
                                                                                test_loader = test_loader,
                                                                                criterion = loss_fn,
                                                                                optimizer = optimizer,
                                                                                device = device,
                                                                                num_epoches = epochs,
                                                                                check_point_filename = checkpoint_file)

print(1)