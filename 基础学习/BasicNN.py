from pylab import *
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

x = torch.linspace(-10, 10, 60)

figure(figsize=(8, 6), dpi=160)
subplot(2, 1, 1)
ax = plt.gca()  # 获取当前的坐标轴
# 边框right、top、bottom、left
ax.spines['right'].set_color('none')  # 右边框为none color
ax.spines['top'].set_color('none')
# 分别把x轴与y轴的刻度设置为bottom与left
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# 分别v把bottom和left类型设置为data，交点为（0，0）
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_position(('data', 0))
# 设置y边界(0, 1)
# plt.ylim((-1, 1))

"""
在sigmoid函数中我们可以看到，其输出是在(0,1)这个开区间，它能够把输入的连续实值变换为0和1之间的输出，
如果是非常大的负数，那么输出就是0；如果是非常大的正数输出就是1，起到了抑制的作用。
但是sigmod由于需要进行指数运算（这个对于计算机来说是比较慢，相比relu），再加上函数输出不是以0为中心的（这样会使权重更新效率降低），
当输入稍微远离了坐标原点，函数的梯度就变得很小了（几乎为零）。
在神经网络反向传播的过程中不利于权重的优化，这个问题叫做梯度饱和，也可以叫梯度弥散。
这些不足，所以现在使用到sigmod基本很少了，基本上只有在做二元分类（0，1）时的输出层才会使用。
"""
sigmod = torch.sigmoid(x)
L1, = plt.plot(x.numpy(), sigmod.numpy(), linewidth=2.5, linestyle="-")

"""
与sigmoid函数类似，当输入稍微远离了坐标原点，梯度还是会很小，但是好在tanh是以0为中心点，如果使用tanh作为激活函数，还能起到归一化（均值为0）的效果。
一般二分类问题中，隐藏层用tanh函数，输出层用sigmod函数，但是随着Relu的出现所有的隐藏层基本上都使用relu来作为激活函数了
"""
tanh = torch.tanh(x)
L2, = plt.plot(x.numpy(), tanh.numpy(), linewidth=2.5, linestyle="-", label='tanh')
plt.legend(handles=[L1, L2], labels=['sigmod', 'tanh'], loc='best')

"""
Relu（Rectified Linear Units）修正线性单元
a=max(0,z) 导数大于0时1，小于0时0。
也就是说： z>0时，梯度始终为1，从而提高神经网络基于梯度算法的运算速度。然而当 z<0时，梯度一直为0。 
ReLU函数只有线性关系（只需要判断输入是否大于0）不管是前向传播还是反向传播，都比sigmod和tanh要快很多

当输入是负数的时候，ReLU是完全不被激活的，这就表明一旦输入到了负数，ReLU就会死掉
但是到了反向传播过程中，输入负数，梯度就会完全到0，这个和sigmod函数、tanh函数有一样的问题。 但是实际的运用中，该缺陷的影响不是很大。
"""
subplot(2, 1, 2)
relu = F.relu(x)
L3, = plt.plot(x.numpy(), relu.numpy())
plt.legend(handles=[L3], labels=['Relu'], loc='best')
plt.xlabel('z')
show()

ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
plt.ylim((-3, 10))
"""
为了解决relu函数z<0时的问题出现了 Leaky ReLU函数，该函数保证在z<0的时候，梯度仍然不为0。 ReLU的前半段设为αz而非0，通常α=0.01  a=max(αz,z)
"""
l_relu = F.leaky_relu(x, 0.1)  # 这里的0.1是为了方便展示，理论上应为0.01甚至更小的值
plt.plot(x.numpy(), l_relu.numpy())
plt.xlabel('z')
show()

# 误差朝w和b的方向减少：梯度下降
# 印象型的解释
# https://zhuanlan.zhihu.com/p/66534632
# https://zhuanlan.zhihu.com/p/65472471