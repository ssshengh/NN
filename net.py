import torch
import torch.nn as nn
import torch.nn.functional as F


# 简单CNN网络
class Net(nn.Module):

    def __init__(self):
        # 搭建卷积神经网络各个层
        super(Net, self).__init__()
        # 两个卷积层和采样层，一个从输入尺度1映射为输出尺度6（channel），下一个从6映射到16，卷积核均为5*5，步长为1，核元素间距1，添加偏置
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层，利用线性函数: y = Wx + b， 就是正常的神经网络
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 数据进入后在各个层处理的过程
    def forward(self, x):
        # 在卷积层后进行池化，池化窗：(2, 2)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 池化窗是正方形，可以直接使用一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 返回一个有相同数据但大小不同的tensor
        x = x.view(-1, self.num_flat_features(x))
        # 进入常规神经网络，激活函数均使用ReLU(The Rectified Linear Unit/修正线性单元)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除批次维度外的所有维度出来
        num_features = 1
        # 每个元素作1的缩放
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# net.parameters()返回可被学习的参数（权重）列表和值
params = list(net.parameters())
print('打印每一层的权重')
print(len(params))
# print(params[0].size())  # 卷积层1的权重
for i in range(0, len(params)):
    print(params[i].size())

# 返回一个张量，包含了从标准正态分布(均值为1，方差为 1，即高斯白噪声)中抽取一组随机数
input1 = torch.randn(1, 1, 32, 32)
print('用一组32*32随机数处理测试:', input1)
out = net(input1)
print('输出为：', out)

# 将所有参数的梯度缓存清零，然后进行随机梯度的的反向传播：
net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input1)
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使target和output的shape相同
criterion = nn.MSELoss()  # 创建一个衡量输入 x ( 模型预测输出 )和目标 y 之间均方误差标准

loss = criterion(output, target)  # 一个损失函数接受一对 (output, target) 作为输入，计算一个值来估计网络的输出和目标值相差多少。
print('损失计算：', loss)
