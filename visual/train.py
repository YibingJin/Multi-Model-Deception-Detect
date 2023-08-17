import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import os
from cnn_gru import Net, simpleGRU, generater, ExpNet
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from dataset_tensor import Mydataset

device = 'cpu'

max_epoch = 100
lr = 8e-4
net = Net()
rnn = simpleGRU()
gen = generater(FC_layers=2, bidirecion=False)
expNet = ExpNet(net, rnn, gen)

criterion = nn.BCELoss()
optimizer = optim.Adam(expNet.parameters(), lr=lr, weight_decay=0.01)

dataset_train = Mydataset(txt_name='name.txt', train=True)
dataset_test = Mydataset(txt_name='name.txt', train=False)
#补零 不同视频长度不一样
def collateFunc(batch):
    Result = []
    Label = np.array([])
    Length = np.array([]) #补零之前视频sequence lenth
    Index=[]
    for result, label, length, index in batch:
        Result.append(result)
        Label = np.append(Label, label)
        Length = np.append(Length, length)
        Index = np.append(Index,index)
    Result = pad_sequence(Result, batch_first=True)
    Label = torch.tensor(Label)
    Length = Length.astype(np.int8)
    Index = Index.astype(np.int8)

    return Result, Label, Length,Index


dataloader_train = DataLoader(dataset=dataset_train, batch_size=8, shuffle=True, drop_last=False, collate_fn=collateFunc)
dataloader_test = DataLoader(dataset=dataset_test, batch_size=len(dataset_test), shuffle=True, drop_last=False, collate_fn=collateFunc)

if (torch.cuda.is_available()):
    device = torch.device("cuda:0")

expNet = expNet.to(device)
criterion = criterion.to(device)

testresult=[]
trainresult=[]
loss_training=[]
for epoch in range(max_epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloader_train):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, lengths,index= data
        inputs, labels= inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = expNet(inputs, lengths)

        onehot_target = torch.eye(2)[labels.long(), :].to(device)


        loss = criterion(outputs, onehot_target)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
           # print every 2000 mini-batches
    if epoch>5 and (epoch+1)%1 == 0:
        PATH = './checkpoint/epoch_{}_add_net.pth'.format(epoch+1)
        torch.save(net.state_dict(), PATH)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            expNet.eval()
            for data in dataloader_test:
                inputs, labels, lengths, index= data
                inputs, labels= inputs.to(device), labels.to(device)

                # forward + backward + optimize
                outputs = expNet(inputs, lengths)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        expNet.train()
        print(f'test Accuracy : {100 * correct / total} %')
        accuracy=100 * correct / total
        testresult.append(accuracy)
        correct1 = 0
        total1 = 0
        with torch.no_grad():
            expNet.eval()
            for data in dataloader_train:
                inputs, labels, lengths,index = data
                inputs, labels = inputs.to(device), labels.to(device)

                # forward + backward + optimize
                outputs = expNet(inputs, lengths)

                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total1 += labels.size(0)
                correct1 += (predicted == labels).sum().item()
        expNet.train()
        #print(f'train Accuracy : {100 * correct1 / total1} %')
        accuracy1 = 100 * correct1 / total1
        trainresult.append(accuracy1)
    print(f'epoch:{epoch + 1} loss: {running_loss / (i+1):.6f}')
    loss_training.append(running_loss)
    running_loss = 0.0

print('Finished Training')

PATH = './checkpoint/add_net.pth'
torch.save(net.state_dict(), PATH)

#############test###############33333
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    expNet.eval()
    for data in dataloader_test :
        inputs, labels, lengths, index = data
        inputs, labels,= inputs.to(device), labels.to(device)

        # forward + backward + optimize
        outputs = expNet(inputs, lengths)

        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('final test')
        print('index',index)
        print('predicted',predicted)
        print('label',labels)
expNet.train()
print(f'Accuracy : {100 * correct / total} %')
print(max(testresult))
plt.figure(1)
plt.plot(trainresult, label='train')
plt.plot(testresult, label='test')
plt.legend()
plt.figure(2)
plt.plot(loss_training,label='loss')
plt.show()
