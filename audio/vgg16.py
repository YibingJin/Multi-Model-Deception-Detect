import torch.nn as nn
import torch

class VGG(nn.Module):  # VGG继承nn.Module(所有神经网络的基类)
    def __init__(self, features, class_num=2, init_weights=False, weights_path=None):
        """生成的网络特征，分类的个数，是否初始化权重，权重初始化路径"""
        super(VGG, self).__init__()  # 多继承
        self.features = features
        self.classifier = nn.Sequential(
            # 将最后三层全连接和分类进行打包
            # torch.nn.Squential：一个连续的容器。模块将按照在构造函数中传递的顺序添加到模块中。或者，也可以传递模块的有序字典。
            nn.Dropout(p=0.5),  # 随机失活一部分神经元，用于减少过拟合，默认比例为0.5，仅用于正向传播。
            nn.Linear(512*7*7, 2048),
            # 全连接层，将输入的数据展开成为一位数组。将512*7*7=25088展平得到的一维向量的个数为2048
            nn.ReLU(True),  # 定义激活函数
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),  # 定义第二个全连接层
            nn.ReLU(True),
            nn.Linear(2048, class_num)  # 最后一个全连接层。num_classes：分类的类别个数。
        )
        if init_weights and weights_path is None:
            self._initialize_weights()  # 是否对网络进行初始化

        if weights_path is not None:
            self.load_state_dict(torch.load(weights_path), strict=False)

    def forward(self, x):
        """
        前向传播,x是input进来的图像
        """
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)  # 进行展平处理，start_dim=1，指定从哪个维度开始展平处理，
        # 因为第一个维度是batch，不需要对它进行展开，所以从第二个维度进行展开。
        # N x 512*7*7
        x = self.classifier(x)  # 展平后将特征矩阵输入到事先定义好的分类网络结构中。
        return x

    def _initialize_weights(self):
        """
        初始化模型权重
        """
        for m in self.modules():  # 用m遍历网络的每一个子模块，即网络的每一层
            if isinstance(m, nn.Conv2d):  # 若m的值为 nn.Conv2d,即卷积层
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)  # 用xavier初始化方法初始化卷积核的权重
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 若偏置不为None，则将偏置初始化为0
            elif isinstance(m, nn.Linear):  # 若m的值为nn.Linear,即池化层
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                # 用一个正太分布来给权重进行赋值，0为均值，0.01为方差
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    """
    提取特征网络结构，
    cfg.list：传入配置变量，只需要传入对应配置的列表
    """
    layers = []  # 空列表，用来存放所创建的每一层结构
    in_channels = 3  # 输入数据的深度，RGB图像深度数为3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            # 若为最大池化层，创建池化操作，并为卷积核大小设置为2，步距设置为2，并将其加入到layers
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
            # 创建卷积操作，定义输入深度，配置变量，卷积核大小为3，padding操作为1，并将Conv2d和ReLU激活函数加入到layers列表中
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg(model_name="vgg16", weights_path=None):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(make_features(cfg), weights_path=weights_path)
    return model

if __name__=='__main__':
    model_name = "vgg16"
    # net = vgg(model_name=model_name, num_classes=6, init_weights=True)
    net = vgg(model_name=model_name)
    a=torch.rand(2, 3, 224, 224)
    outputs = net(a)
    print(outputs)
