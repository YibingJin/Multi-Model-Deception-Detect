import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
import hiddenlayer as h
from torchviz import make_dot
import tensorwatch as tw
class Net(nn.Module):
    def __init__(self,):
        super(Net,self).__init__()
        self.conv1 = nn.Conv1d(in_channels=43, out_channels=64, kernel_size=3, stride=1,
                               padding=0)  #14
        self.maxpool1 = nn.MaxPool1d(5, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.maxpool2=nn.MaxPool1d(2,stride=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0)  # 3
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        self.activate = nn.ReLU()
        self.FC1 = nn.Linear(1152, 128)
        self.FC2 = nn.Linear(512,256)
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(64)
        self.batchnorm3 = nn.BatchNorm1d(128)



    def forward(self, x):
        x = self.conv1(x)

        x = self.batchnorm1(x)
        x = F.relu(x)
        #print('conv1',x.shape)
        x = self.maxpool1(x)
        #print('maxpool1',x.shape)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.relu(x)
        #print('conv2', x.shape)
        x = self.maxpool2(x)
        #print('maxpool2', x.shape)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.relu(x)
        #print('conv3', x.shape)
        #x = F.relu(self.conv4(x))
        #print('conv4', x.shape)
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #print('flatten',x.shape)
        x = F.relu(self.FC1(x))
        #print('linear',x.shape)
        x = self.dropout(x)
        # x = F.relu(self.FC2(x))
        #print('dropout',x.shape)
        return x




class simpleGRU(nn.Module):
    def __init__(self, input_dim=128, hid_dim=128, num_layers=1, bidirection=False):
        super(simpleGRU, self).__init__()
        self.rnn = nn.GRU(input_dim, hid_dim, num_layers=num_layers, batch_first=True,
                          bidirectional=bidirection)

    def forward(self, input):
        out = self.rnn(input)
        #print('gru')
        #print(len(out))
        #print(out[0].shape)
        #print(out[1].shape)
        return out

class generater(nn.Module):
    def __init__(self, FC_layers, bidirecion):
        super(generater, self).__init__()

        if bidirecion:
            if FC_layers == 2:
                self.blk1 = nn.Sequential(
                    nn.Linear(128, 16),
                    nn.ReLU(),
                    nn.Dropout(p=0.4),
                    nn.Linear(16, 2),
                    torch.nn.Sigmoid()

                )

        else:
            if FC_layers == 2:
                self.blk1 = nn.Sequential(
                    nn.Linear(128, 16),
                    nn.ReLU(),
                    nn.Dropout(p=0.4),
                    nn.Linear(16, 2),
                    torch.nn.Sigmoid()
                )


    def forward(self, input):
        return self.blk1(input)

class ExpNet(nn.Module):
    def __init__(self, Net, rnn, generater):
        super(ExpNet, self).__init__()

        self.Net = Net
        self.rnn = rnn
        self.generater = generater

    def forward(self, input, lengths):
        bth_size, seq_len, fea_size, emb_dim = input.shape
        input = input.reshape(bth_size*seq_len, fea_size, emb_dim)
        input = input.permute(0, 2, 1) #使卷积核（感受野）每次处理的信息是完整的，
        input = self.Net(input)
        input = input.reshape(bth_size, seq_len, -1)
        #print('gru input',input.shape)
        # mixfeature = pack_padded_sequence(input, lengths, batch_first=True, enforce_sorted=False)
        rnn_out, _ = self.rnn(input)
        # print('rnn_out')
        #print('gruout',rnn_out.shape)
        result = torch.tensor([], device=rnn_out.device)
        #print('length',lengths.shape)
        #print('length', lengths)
        for i in range(len(lengths)):
            #print('lengths[i]',lengths[i])
            result = torch.cat((result, rnn_out[i, lengths[i]-1, :].unsqueeze(dim=0)), dim=0)
            #数据被补过零
        # print('result')
        #print('gru output reshapre',result.shape)
        FC_out = self.generater(result)

        # print(FC_out.shape)

        return FC_out


if __name__=='__main__':
    net = Net()
    rnn = simpleGRU()
    gen = generater(FC_layers=2, bidirecion=False)
    expNet = ExpNet(net, rnn, gen)
    a=torch.randn(32, 18, 30, 43)
    c=torch.randn(500, 43, 30)
    b=torch.randn(32,94)
    lengths = torch.from_numpy(np.random.randint(1, 18, 32))
    expNet(a, lengths)
    # vis_graph = h.build_graph(net, torch.randn([500, 43, 30]))  # 获取绘制图像的对象
    # vis_graph.theme = h.graph.THEMES["blue"].copy()  # 指定主题颜色
    # vis_graph.save("./demo2.png")  # 保存图像的路径
    # d = net(c)
    # MyConvNetVis = make_dot(d, params=dict(list(net.named_parameters()) + [('x', c)]))
    # MyConvNetVis.format = "png"
    # # 指定文件生成的文件夹
    # MyConvNetVis.directory = "data"
    # # 生成文件
    # MyConvNetVis.view()


