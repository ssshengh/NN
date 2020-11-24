"""
logistic回归是一种广义线性回归（generalized linear model），与多重线性回归分析有很多相同之处。
它们的模型形式基本上相同，都具有 wx + b，其中w和b是待求参数，其区别在于他们的因变量不同，多重线性回归直接将wx+b作为因变量，即y =wx+b,
而logistic回归则通过函数L将wx+b对应一个隐状态p，p =L(wx+b),然后根据p 与1-p的大小决定因变量的值。如果L是logistic函数，就是logistic回归，如果L是多项式函数就是多项式回归。
说的更通俗一点，就是logistic回归会在线性回归后再加一层logistic函数的调用。
logistic回归主要是进行二分类预测，我们在激活函数时候讲到过 Sigmod函数，Sigmod函数是最常见的logistic函数，
因为Sigmod函数的输出的是是对于0~1之间的概率值，当概率大于0.5预测为1，小于0.5预测为0。
"""

import torch
import torch.nn as nn
import numpy as np


# logisit model
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(24, 2)  # 由于24个维度已经固定了，所以这里写24

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


# 测试集上的准确率
def test(pred, lab):
    # 1&-1 is max of row, 0 is max of column
    # [0] indicates vector of max values, while [1] indicates the location of max values
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())  #


"""
数据集：German Credit数据是根据个人的银行贷款信息和申请客户贷款逾期发生情况来预测贷款违约倾向的数据集，
数据集包含24个属性，1000条数据。其中每个属性下面还有许多的选项，由离散化等方法处理，最后标签为1或者2，1代表好，2代表坏
"""
data = np.loadtxt("./data/germon/german.data-numeric")

n, l = data.shape  # row, column
#   Z-score标准化方法， 这种方法给予原始数据的均值（mean）和标准差（standard deviation）进行数据的标准化。
for j in range(l - 1):  # loop by column
    meanVal = np.mean(data[:, j])  # 第i列均值
    stdVal = np.std(data[:, j])  # numpy.std() 求标准差的时候默认是除以 n 的，即是有偏的，np.std无偏样本标准差方式为加入参数 ddof = 1；
    data[:, j] = (data[:, j] - meanVal) / stdVal

np.random.shuffle(data)  # 打乱数据

"""
900条用于训练，100条作为测试
"""
train_data = data[:900, :l - 1]
train_lab = data[:900, l - 1] - 1  # 0 good ; 1 bad
test_data = data[900:, :l - 1]
test_lab = data[900:, l - 1] - 1

net = LR()
criterion = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss损失
optm = torch.optim.Adam(net.parameters())  # Adam优化
epochs = 10000  # 训练1000次

for i in range(epochs):
    # 指定模型为训练模式，计算梯度
    net.train()
    # 输入值都需要转化成torch的Tensor
    x = torch.from_numpy(train_data).float()
    y = torch.from_numpy(train_lab).long()
    y_hat = net(x)
    loss = criterion(y_hat, y)  # 计算损失
    optm.zero_grad()  # 前一步的损失清零
    loss.backward()  # 反向传播
    optm.step()  # 优化
    if (i + 1) % 100 == 0:  # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in = torch.from_numpy(test_data).float()
        test_l = torch.from_numpy(test_lab).long()
        test_out = net(test_in)
        # 使用我们的测试函数计算准确率
        accu = test(test_out, test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accu))
