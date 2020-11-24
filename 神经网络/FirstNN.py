# %% 读入数据
# from pylab import *
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import re

data_path = '/home/shenheng/PycharmProjects/pythonProject/神经网络/data/test.csv'
rides = pd.read_csv(data_path)
# %% 简单的看看数据
rides.head()
# data = rides.iloc[0: 958, :]  # 忽略地点作为练手，以初始自行车数量减去最后的自行车数量作为车辆变化 取出一月一日的
data = rides.loc[rides['Start station'] == '5th & F St NW']
# data = rides['Start date'].map(lambda x: re.findall(r'2011-01-01', str(x)))
StartNum = data['Start station number']
EndNum = data['End station number']
cnt = StartNum - EndNum

series = np.arange(len(cnt))  # 等差数列
num = np.array(cnt)

# %% 画图来看看车辆数的差值每天是如何变化的
plt.figure(figsize=(10, 7))
plt.plot(series, num, 'o-')
plt.xlabel('time interval')
plt.ylabel('cycle number difference')
plt.show()

# %%
# 测试数据和训练数据
train = num[0:50]
test = num[50:110]
train_series = torch.FloatTensor(np.arange(len(train), dtype=float) / len(train))  # 1,2,3,4,....
train_num = torch.FloatTensor(np.array(train, dtype=float))
y = train_num.view(50, -1)
x = train_series.view(50, -1)

sz = 10  # 隐藏层神经元数量
weights = torch.randn((1, sz), requires_grad=True)  # 输入层到隐藏层权重矩阵
biases = torch.randn((sz), requires_grad=True)  # 输入层到隐藏层偏置
eights1 = torch.randn((sz, 1), requires_grad=True)  # 隐藏层到输出权重

learing_rate = 0.001  # 学习率
losses = []  # 记录每一次迭代的损失函数

# %%
epochs = 100000  # 训练100000次
for i in range(epochs):
    # 输入层到隐藏层计算
    hidden = x * weights + biases
    # hidden (50,10) 50个数据点，10个隐藏层神经元

    hidden = torch.sigmoid(hidden)  # 激活
    # 隐藏层到输出层
    predictions = hidden.mm(eights1)
    # predictions (50,1)
    # 预测输出与实际单车数量差做均方误差
    loss = torch.mean((predictions - y) ** 2)  # loss为一个标量
    losses.append(loss)

    if i % 10000 == 0:
        print('loss: ', loss)

    # *****************************************
    # 梯度下降，误差反向传播
    loss.backward()

    # 利用上一步得到的参数进行预测
    weights.data.add_(- learing_rate * weights.grad.data)
    biases.data.add_(- learing_rate * biases.grad.data)
    eights1.data.add_(- learing_rate * eights1)

    # 清空梯度
    weights.grad.data.zero_()
    biases.grad.data.zero_()
    weights.grad.data.zero_()

# %% 看一下误差
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

# %%
x_data = x.data.numpy()
plt.figure(figsize=(10, 7))
xplot, = plt.plot(x_data, y.data.numpy(), 'o')
yplot, = plt.plot(x_data, predictions.data.numpy())
plt.xlabel('x')
plt.ylabel('cycle Num')
plt.legend([xplot, yplot], ['data', 'Predictions'])
plt.show()
