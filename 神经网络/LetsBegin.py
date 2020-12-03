# %% 读入数据
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import re

data_path = '/home/shenheng/PycharmProjects/pythonProject/神经网络/data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
rides = pd.read_csv(data_path)
# %%
read = rides[
    ['Flow Bytes/s', ' Flow Packets/s', 'Total Length of Fwd Packets', ' Bwd Packet Length Mean', ' Flow IAT Min',
     ' Label']]
read = read.round(4)
# %%
# 用apply（）直接处理一整个列对一整个列进行赋值
#
read.loc[read[' Label'] == 'BENIGN', ' Label'] = 1  # secure
read.loc[read[' Label'] == 'DDoS', ' Label'] = 0  # under attack
a = 1000
read.loc[:, 'Flow Bytes/s'] = read['Flow Bytes/s'] / a
read.loc[:, ' Flow Packets/s'] = read[' Flow Packets/s'] / a
# 将某1列（series格式）中的 inf 替换为数值。
# df['Col'][np.isinf(df['Col'])] = -1
# df['Col'][np.isinf(df['Col'])] = np.nan
# df.replace(np.inf, -1) #替换正inf为-1
# df.replace([np.inf, -np.inf], np.nan) #替换正负inf为NAN

read = read.replace(np.inf, 0)
read = read.replace(np.nan, 0)
# %%
mean = read['Flow Bytes/s'].mean(skipna=True)

# %% 处理变量，标准化 [-1, 1]  expect 'Flow Bytes/s', ' Flow Packets/s',
quant_features = ['Flow Bytes/s', ' Flow Packets/s', 'Total Length of Fwd Packets', ' Bwd Packet Length Mean',
                  ' Flow IAT Min']
scaled_features = {}  # 储存每一个变量的均值和方差
for each in quant_features:
    # z-score
    # mean, std = read[each].mean(), read[each].std()
    # scaled_features[each] = [mean, std]
    # # 对每一个进行标准化
    # read.loc[:, each] = (read[each] - mean) / std

    # min-max标准化
    max1, min1 = read[each].max(), read[each].min()
    scaled_features[each] = [max1, min1]
    read.loc[:, each] = (read[each] - min1) / (max1 - min1)

# %% 看一看
FlowBy = read['Flow Bytes/s']
FlowPa = read[' Flow Packets/s']
FwdPack = read['Total Length of Fwd Packets']
BwdPack = read[' Bwd Packet Length Mean']
FlowIat = read[' Flow IAT Min']
Lab = read[' Label']
plt.figure(figsize=(10, 7), dpi=160)

num = np.arange(len(Lab))
Flow_Byte = np.array(FlowBy)
Flow_Packets = np.array(FlowPa)
Fwd_pack = np.array(FwdPack)
Bwd_pack = np.array(BwdPack)
Flow_iat = np.array(FlowIat)
Label = np.array(Lab)

Fb, = plt.plot(num, Fwd_pack, '-g')
Fp, = plt.plot(num, Bwd_pack, '-b')
L, = plt.plot(num, Label, 'o')
plt.xlabel('packet ID')
plt.ylabel('Byte|bite/s')


plt.legend([Fb, Fp, L], ['Flow Bytes', 'Flow Packets', 'attack'])
plt.savefig("FlowAbout.png", dpi=72)
plt.show()


# %%
plt.figure(figsize=(10, 7), dpi=160)
Fb1, = plt.plot(num, Flow_Byte, '-g')
Fp1, = plt.plot(num, Flow_Packets, '-b')
L, = plt.plot(num, Label, 'o')
plt.xlabel('packet ID')
plt.ylabel('Byte|bite/s')

plt.legend([Fb1, Fp1, L], ['Flow Bytes', 'Flow Packets', 'attack'], loc='upper left')
plt.savefig("Fwd&Bwd.png", dpi=72)
plt.show()
# Flow Iat 过于稀疏，不作为考虑
# %%
# 处理变量
test_data = read[-25744:]
train_data = read[:-25744]

# 目标列包含的字段
target = ' Label'
# 训练集与测试集把变量与Label分离
features, targets = train_data.drop(target, axis=1), train_data[target]
test_features, test_tar = test_data.drop(target, axis=1), test_data[target]
# to numpy
X = features.values
Y = targets.values
Y = Y.astype(float)

Y = np.reshape(Y, [len(Y), 1])
losses = []

# %% NN
input_size = features.shape[1]  # 查看数据类型可以看见是20w
hidden_size = 50  # 隐藏层数量
output_size = 1  # 1个输出层
batch_size = 10000  # 批处理，每一批大小

neu = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size),
    torch.nn.Sigmoid(),
    torch.nn.Linear(hidden_size, output_size)
)

cost = torch.nn.MSELoss()  # 损失
optimizer = torch.optim.SGD(neu.parameters(), lr=0.01)  # 优化

# %%
epoch = 1000
for i in range(epoch):
    batch_loss = []
    # 10000个一批
    for start in range(0, len(X), batch_size):
        end = start + batch_size if start + batch_size < len(X) else len(X)
        xx = torch.FloatTensor(X[start:end])
        yy = torch.FloatTensor(Y[start:end])
        predict = neu(xx)
        loss = cost(predict, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(i, np.mean(batch_loss))

# %% 看一下损失曲线
plt.plot(np.arange(len(losses)) * 100, losses)
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.savefig("Loss.png", dpi=72)
plt.show()
# %%
# 处理测试集标签
traget_test = test_tar.values.reshape([len(test_tar), 1])
traget_test = traget_test.astype(float)
# 将测试集扔进去进行预测
x_test = torch.FloatTensor(test_features.values)
y_test = torch.FloatTensor(traget_test)

predict_test = neu(x_test) #预测结果
predict_test = predict_test.data.numpy()

# %%
fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(predict_test, Label='Prediction')
ax.plot(traget_test, Label='Data')
plt.legend(['Prediction', 'Data'], loc='upper left')
plt.savefig("prediction&real.png", dpi=72)
plt.show()
