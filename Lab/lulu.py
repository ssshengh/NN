from sklearn.preprocessing import StandardScaler  # 通过减去均值并除以单位方差来标准化数据
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt


# %%
def dataSets(path: str):
    """
    读取文件，转换为DataFrame，并且得到归一化后的数据
    :param test_size: 测试集大小，定义方式应该为这样：test_size = int(len(data_all) // 10)
    :param path: 数据储存路径，需要为csv文件
    :return: 返回归一化后的数据,包括数据集、标签、测试集数据、标签、校验集数据、标签，均为dataFrame
    """
    data_read = pd.read_csv(path)
    data_read.columns = ['dertapx', 'dertapy', 'dertavx', 'dertavy', 'ifattacker']
    data_read = data_read.sample(frac=1).reset_index(drop=True)  # 数据扰乱
    data_read = data_read.dropna(how='any')  # 丢掉所有的nan数据
    scaler = StandardScaler()  # fit求均值方差，transform求归一化数据
    target_all = data_read['ifattacker'].to_numpy()  # 获取标签
    data_all = data_read.drop(['ifattacker'], axis=1)  # 获取所有数据，是一个4列的数据
    # 归一化处理结束
    data_all = scaler.fit_transform(data_all)
    data_all = pd.DataFrame(data_all)

    test_size = int(len(data_all) // 10)  # 数据分割数量
    # 训练集
    train_data = data_all[2 * test_size:]
    train_label = target_all[2 * test_size:]

    test_data = data_all[test_size: 2 * test_size]
    test_label = target_all[test_size: 2 * test_size]

    valid_data = data_all[: test_size]
    valid_label = target_all[: test_size]
    return train_data, train_label, test_data, test_label, valid_data, valid_label


# %%
def NN(input_size: int, hidden_size: int, output_size: int):
    """
    一维神经网络模型建模，主要是一个隐层，一个ReLU层，一个SoftMax层
    :param input_size: 输入数据维度
    :param hidden_size: 隐层维度
    :param output_size: 输出维度，如果是二分类的话为2
    :return: 一个神经网络模型neu，损失函数和优化器
    """

    neu = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_size),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_size, output_size),
        torch.nn.LogSoftmax(dim=1)
    )
    cost = torch.nn.NLLLoss()  # 损失函数
    optimizer = torch.optim.SGD(neu.parameters(), lr=0.001)  # 优化算法
    return neu, cost, optimizer


# %%
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size(数据量)行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    # 比如三维张量，第一个是z，第二个是行第三个是列。出来的是一个两列的矩阵，第一列是纬度上的最值，第二列是维度上的位置
    rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    #  view_as(tensor)返回被视作与给定的tensor相同大小的原tensor。
    return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素›


# %%
def run(input_size: int, hidden_size: int, output_size: int, epoch: int, path: str):
    train_data, train_label, test_data, test_label, valid_data, valid_label = dataSets(path)  # 获取数据
    neu, cost, optimizer = NN(input_size, hidden_size, output_size)  # 获取神经网络
    losses = []
    records = []
    epoch_size = 10  # 十个epoch
    a = zip(train_data.values, train_label)
    for epoch in range(epoch_size):
        # print(str(epoch) + "次：")
        for i, data in enumerate(zip(train_data.values, train_label)):
            x, y = data  # x为特征，y为标签
            x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)  # 重新构造形状，1行，-1列指从行来判断
            y = torch.tensor(np.array([y]), dtype=torch.long)
            # 清空梯度
            optimizer.zero_grad()
            # 模型预测
            predict = neu(x)
            # 计算损失函数
            loss = cost(predict, y)
            # 将损失函数数值加入到列表中
            losses.append(loss.data.numpy())
            # 开始进行梯度反传
            loss.backward()
            # 开始对参数进行一步优化
            optimizer.step()

            if i % 3000 == 0:
                val_losses = []
                rights = []
                # 在所有校验数据集上实验
                # dataframe必须要取出values才能得到正确结果，不然的话得到的就是列数
                for j, val in enumerate(zip(valid_data.values, valid_label)):
                    x, y = val
                    x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                    y = torch.tensor(np.array([y]), dtype=torch.long)
                    predict = neu(x)
                    # 调用rightness函数计算准确度,对一个数据进行比较，正确得到1，错误得到0
                    # 只输出一个的话，第一个为代表是否正确的0，1；第二个为代表比较了几个的1
                    right = rightness(predict, y)
                    rights.append(right)  # rights里面是一个0101的数组，对应于校验集上数据的正确性
                    loss1 = cost(predict, y)
                    val_losses.append(loss1.data.numpy())
                # 因此可以计算校验集上的正确率
                right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
                # 注意校验损失和训练损失为其中的均值
                print('第{}轮——损失为交叉熵均值：训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                                      np.mean(val_losses), right_ratio))
                records.append([np.mean(losses), np.mean(val_losses), right_ratio])  # 记录下三者
    return records, neu


# %%

re, model1 = run(4, 10, 2, 10, '/Users/shenheng/Code/NN/Lab/data/lulu.csv')


# %%
def test(test_data: pd.DataFrame, test_label, model):
    # 在测试集上分批运行，并计算总的正确率
    vals = []  # 记录准确率所用列表

    # 对测试数据集进行循环
    for data, target in zip(test_data.values, test_label):
        data = torch.tensor(data, dtype=torch.float).view(1, -1)
        target = torch.tensor(np.array([target]),dtype=torch.long)
        output = model(data)  # 将特征数据喂入网络，得到分类的输出
        val = rightness(output, target)  # 获得正确样本数以及总样本数
        vals.append(val)  # 记录结果
    # 计算准确率
    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    right_rate = 1.0 * rights[0].data.numpy() / rights[1]
    print("测试集正确率为：" + str(right_rate))
    return vals
