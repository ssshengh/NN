# 首先要引入相关的包
import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.datasets as datasets


# Dataset是一个抽象类，为了能够方便的读取，需要将要使用的数据包装为Dataset类。 自定义的Dataset需要继承它并且实现两个成员方法
# __getitem__() 该方法定义用索引(0 到 len(self))获取一条数据或一个样本
# __len__() 该方法返回数据集的总长度

# 定义一个数据集
class BulldozerDataset(Dataset):
    """ 数据集演示 """

    def __init__(self, csv_file):
        """实现初始化方法，在初始化的时候将数据读载入"""
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        """
        返回df的长度
        iloc比较简单，它是基于索引位来选取数据集，【0:4】就是选取 0，1，2，3这四行，需要注意的是这里是前闭后开集合
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        根据 idx 返回一行数据
        """
        if idx == 0:
            return self.df.SalesID
        elif idx == 1:
            return self.df.SalePrice
        else:
            print("Wrong Index!!")
        # return self.df.iloc[idx]


# 读入两列的数据
ds_demo = BulldozerDataset('median_benchmark.csv')

# 实现了 __len__ 方法所以可以直接使用len获取数据总数
print('length:', len(ds_demo))
# 用索引可以直接访问对应的数据，对应 __getitem__ 方法
# print(ds_demo[0][1])

"""
DataLoader为我们提供了对Dataset的读取操作，
常用参数有：batch_size(每个batch的大小)、 shuffle(是否进行shuffle操作，打乱数据)、 num_workers(加载数据的时候使用几个子进程)。下面做一个简单的操作
DataLoader返回的是一个可迭代对象，我们可以使用迭代器分次获取数据
"""
dl = torch.utils.data.DataLoader(ds_demo[0], batch_size=10, shuffle=True, num_workers=0)
# 我们可以使用迭代器分次获取数据
idata = iter(dl)
print('第一个数据区域：', next(idata))
print('第二个数据区域：', next(idata))
# 常见的用法是使用for循环对其进行遍历
for i, data in enumerate(dl):  # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    print(f'第{i}个数据区域', i, data)
    # 为了节约空间，这里只循环一遍
    if i == 2:
        break

"""
torchvision 包
torchvision 是PyTorch中专门用来处理图像的库，PyTorch官网的安装教程中最后的pip install torchvision 就是安装这个包。
torchvision.datasets 可以理解为PyTorch团队自定义的dataset，这些dataset帮我们提前处理好了很多的图片数据集，我们拿来就可以直接使用
"""

trainset = datasets.MNIST(root='./data',  # 表示 MNIST 数据的加载的目录
                          train=True,  # 表示是否加载数据库的训练集，false的时候加载测试集
                          download=True,  # 表示是否自动下载 MNIST 数据集
                          transform=None)  # 表示是否需要对数据进行预处理，none为不进行预处理
print(trainset.data)