import os
import PIL
import numpy as np
import skimage.io
import skimage.transform
import matplotlib as mpl
import torch,torchvision
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data_path: str, train=True, transform=None):
        self.data_path = data_path
        self.train_flag = train
        if transform is None:
            self.transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.Resize(size=(224, 224)),  # 尺寸规范
                    torchvision.transforms.CenterCrop(224),
                    torchvision.transforms.ToTensor(),  # 转化为tensor
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
                ])
        else:
            self.transform = transform
        self.path_list = os.listdir(data_path)  # 列出所有图片命名


    def __getitem__(self, idx: int):

        img_path = self.path_list[idx]
        # 例如 img_path 值 cat.10844.jpg -> label = 0
        if img_path.split('_')[1] == 'lie':
            label = 1
        else:
            label = 0
        label = torch.tensor(label, dtype=torch.int64)  # 把标签转换成int64
        img_path = os.path.join(self.data_path, img_path)  # 合成图片路径
        img = PIL.Image.open(img_path)  # 读取图片
        img = self.transform(img)  # 把图片转换成tensor
        return img, label

    def __len__(self) -> int:
        return len(self.path_list)  # 返回图片数量

train_path = './processedaudio/training set'
test_path = './processedaudio/test set'
train_datas = MyDataset(train_path)
test_datas = MyDataset(test_path)

train_loader = torch.utils.data.DataLoader(train_datas, batch_size=16,
                                            shuffle=True, pin_memory=True, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_datas, batch_size=16,
                                            shuffle=False, pin_memory=True, num_workers=0)

# for val_data in test_loader:
#     """
#     遍历验证集；将数据划分为图片和对应的标签值
#     对结点进行参数的更新；指认到设备上并传入网络得到输出
#     """
#     val_images, val_labels = val_data
#     length = val_images.size()[0]
#     for i in range(length):
#         Img_PIL_Tensor = val_images[i]
#         new_img_PIL = torchvision.transforms.ToPILImage()(Img_PIL_Tensor).convert('RGB')
#         plt.imshow(new_img_PIL)
#         plt.title(i)
#         plt.show()



