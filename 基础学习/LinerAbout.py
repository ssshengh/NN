import torch
from torch.nn import Linear, Module, MSELoss
from torch.optim import SGD
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import *

#  线性回归 （Linear Regreesion）
"""
线性回归是利用数理统计中回归分析，来确定两种或两种以上变量间相互依赖的定量关系的一种统计分析方法，运用十分广泛。其表达形式为y = w'x+e，e为误差服从均值为0的正态分布。
回归分析中，只包括一个自变量和一个因变量，且二者的关系可用一条直线近似表示，这种回归分析称为一元线性回归分析。如果回归分析中包括两个或两个以上的自变量，且因变量和自变量之间是线性关系，则称为多元线性回归分析。 摘自百度百科
简单的说： 线性回归对于输入x与输出y有一个映射f，y=f(x),而f的形式为aX+b。其中a和b是两个可调的参数，我们训练的时候就是训练a，b这两个参数。
"""

x = np.linspace(0, 20, 500)
y = 5 * x + 7

figure(figsize=(8, 6), dpi=80)
subplot(2, 1, 1)
plt.plot(x, y, color="blue", linewidth=2.5, linestyle="-", label="LinerFunction")
# show()


# 下面生成一些随机的点，来作为训练数据
"""np的随机数用法
np.random.rand
通过本函数可以返回一个或一组服从“0~1”均匀分布的随机样本值。随机样本取值范围是[0,1)，不包括1。 
应用：在深度学习的Dropout正则化方法中，可以用于生成dropout随机向量（dl），例如（keep_prob表示保留神经元的比例）：dl = np.random.rand(al.shape[0],al.shape[1]) < keep_prob
np.random.randn(d0,d1,d2……dn) 
通过本函数可以返回一个或一组服从标准正态分布的随机样本值。
1)当函数括号内没有参数时，则返回一个浮点数； 
2）当函数括号内有一个参数时，则返回秩为1的数组，不能表示向量和矩阵； 
3）当函数括号内有两个及以上参数时，则返回对应维度的数组，能表示向量或矩阵； 
4）np.random.standard_normal（）函数与np.random.randn()类似，但是np.random.standard_normal（）的输入参数为元组（tuple）. 
5)np.random.randn()的输入通常为整数，但是如果为浮点数，则会自动直接截断转换为整数
"""
x = np.random.rand(256)  # 256*1
noise = np.random.randn(256) / 4
y = x * 5 + 7 + noise
df = pd.DataFrame()
df['x'] = x
df['y'] = y

sns.lmplot(x='x', y='y', data=df, aspect=1.5)  # 会自动回归
show()

# 对输入数据做线性变换： $y=w * x+b，其中w代表权重， b代表偏置， 其中参数(1, 1)代表输入输出的特征(feature)数量都是1
model = Linear(1, 1)
# 损失函数我们使用均方损失函数
criterion = MSELoss()

# 优化器我们选择最常见的优化方法 SGD，就是每一次迭代计算 mini-batch 的梯度，然后对参数进行更新，学习率 0.01
optim = SGD(model.parameters(), lr=0.01)
epochs = 3000  # 训练3000次
# 准备训练数据: x_train, y_train 的形状是 (256, 1)， 代表 mini-batch 大小为256， feature 为1. astype('float32') 是为了下一步可以直接转换为 torch.float.
x_train = x.reshape(-1, 1).astype('float32')  # 1-Dimension
y_train = y.reshape(-1, 1).astype('float32')

# 开始训练
for i in range(epochs):
    # 整理输入和输出的数据，这里输入和输出一定要是torch的Tensor类型
    inputs = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    # 使用模型进行预测
    outputs = model(inputs)
    # 梯度置0，否则会累加
    optim.zero_grad()
    # 计算损失
    loss = criterion(outputs, labels)
    # 反向传播
    loss.backward()
    # 使用优化器默认方法优化
    optim.step()
    if i % 100 == 0:
        # 每 100次打印一下损失函数，看看效果
        print('epoch {}, loss {:1.4f}'.format(i, loss.data.item()))
    # if loss.data.item() <= 0.6:
    #     break

# 训练完成了，看一下训练的成果是多少。用 model.parameters() 提取模型参数。 w，b是我们所需要训练的模型参数 我们期望的数据 w=5，b=7可以做一下对比
[w, b] = model.parameters()
print(w.item(), b.item())

# 再次可视化一下我们的模型，看看我们训练的数据
predicted = model.forward(torch.from_numpy(x_train)).data.numpy()
plt.plot(x_train, y_train, 'go', label='data', alpha=0.3)
plt.plot(x_train, predicted, label='predicted', alpha=1)
plt.legend()
plt.show()

# lr参数为学习率，对于SGD来说一般选择0.1 0.01.0.001，如何设置会在后面实战的章节中详细说明
# 如果设置了momentum，就是带有动量的SGD，可以不设置
# 随机梯度下降算法，带有动量（momentum）的算法作为一个可选参数可以进行设置，样例如下：
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# 除了以上的带有动量Momentum梯度下降法外，RMSprop（root mean square prop）也是一种可以加快梯度下降的算法，
# 利用RMSprop算法，可以减小某些维度梯度更新波动较大的情况，使其梯度下降的速度变得更快
# 我们的课程基本不会使用到RMSprop所以这里只给一个实例
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99)

# Adam 优化算法的基本思想就是将 Momentum 和 RMSprop 结合起来形成的一种适用于不同深度学习结构的优化算法
# 这里的lr，betas，还有eps都是用默认值即可，所以Adam是一个使用起来最简单的优化方法
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
