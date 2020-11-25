# %%
import torch
import torch.nn as nn
import torch.optim

# about NLP
import re  # 正则表达式
import jieba  # 结巴中文分词器
from collections import Counter  # 搜集器，统计词频

import matplotlib as plt1
import numpy as np
import matplotlib.pyplot as plt

good_file = '/home/shenheng/PycharmProjects/pythonProject/神经网络/data/NPL/jd_good.txt'
bad_file = '/home/shenheng/PycharmProjects/pythonProject/神经网络/data/NPL/jd_bad.txt'


# %%
# 将文本中的标点符号过滤掉
def filter_punc(sentence1):
    sentence1 = re.sub("[\s+\.\!\/_,$%^*(+\"\'“”《》?“]+|[+——！，。？、丶~@#￥%……&*（）：]+|[a-zA-Z0-9]+]", "", sentence1)
    return sentence1


# 扫描所有的文本，分词、建立词典，分出正向还是负向的评论，is_filter可以过滤是否筛选掉标点符号
def Prepare_data(good_file, bad_file, is_filter=True):
    all_words = []  # 存储所有的单词
    pos_sentences = []  # 存储正向的评论
    neg_sentences = []  # 存储负向的评论
    """
    使用with处理的对象必须有__enter__()和__exit__()这两个方法,通过__enter__方法初始化，然后在__exit__中做善后以及处理异常。
    with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，比如文件使用后自动关闭、线程中锁的自动获取和释放等。
    """
    with open(good_file, 'r') as fr:
        """
        读取文件，
        每一行过滤标点符号，并调用jieba分词
        然后，把分词完成后的一行加入all_words,把单词一个一个的放入pos_sentences
        """
        for idx, line in enumerate(fr):
            # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
            # 可以看见读入的文本是一行一行读入的
            if is_filter:
                # 过滤标点符号
                line = filter_punc(line)
            # 精确分词 line为一个字符串，分词后得到列表作为输出
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                pos_sentences.append(words)
                # 使用append实际是修改一个列表，使用+实际是创建一个新的列表
    print('{0} 包含 {1} 行, {2} 个词.'.format(good_file, idx + 1, len(all_words)))

    count = len(all_words)
    with open(bad_file, 'r') as fr:
        for idx, line in enumerate(fr):
            if is_filter:
                line = filter_punc(line)
            words = jieba.lcut(line)
            if len(words) > 0:
                all_words += words
                neg_sentences.append(words)
    print('{0} 包含 {1} 行, {2} 个词.'.format(bad_file, idx + 1, len(all_words) - count))

    # 建立词典，diction的每一项为{w:[id, 单词出现次数]}
    diction = {}
    cnt = Counter(all_words)
    for word, freq in cnt.items():  # 方法以列表形式（并非直接的列表，若要返回列表值还需调用list函数）返回可遍历的(键, 值) 元组数组。
        diction[word] = [len(diction), freq]  # id是词的长度，从0开始慢慢加
    print('字典大小：{}'.format(len(diction)))
    return pos_sentences, neg_sentences, diction


# 根据单词返还单词的编码
def word2index(word, diction):
    if word in diction:
        value = diction[word][0]
    else:
        value = -1
    return value


# 根据编码获得单词
def index2word(index, diction):
    for w, v in diction.items():
        if v[0] == index:
            return w
    return None


pos_sentences, neg_sentences, diction = Prepare_data(good_file, bad_file, True)
st = sorted([(v[1], w) for w, v in diction.items()])


# list 的 sort 方法返回的是对已经存在的列表进行操作，而内建函数 sorted 方法返回的是一个新的 list，而不是在原来的基础上进行的操作。
# %%词袋模型 目的是按照每个词排列的位置对一句话进行编号

# 输入一个句子和相应的词典，得到这个句子的向量化表示
# 向量的尺寸为词典中词汇的个数，i位置上面的数值为第i个单词在sentence中出现的频率
def sentence2vec(sentence, dictionary):
    vector = np.zeros(len(dictionary))
    for l in sentence:
        vector[l] += 1
    return (1.0 * vector / len(sentence))


# 遍历所有句子，将每一个词映射成编码
dataset = []  # 数据集
labels = []  # 标签
sentences = []  # 原始句子，调试用
# 处理正向评论
for sentence in pos_sentences:
    # 一句话
    new_sentence = []
    for l in sentence:
        # 一个词
        if l in diction:
            # 获取一句话里每个单词的编号，然后放入new_sentence组成一个新的列表
            new_sentence.append(word2index(l, diction))
    # 按照每个词排列的位置对一句话进行编号
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(0)  # 正标签为0
    sentences.append(sentence)

# 处理负向评论
for sentence in neg_sentences:
    new_sentence = []
    for l in sentence:
        if l in diction:
            new_sentence.append(word2index(l, diction))
    dataset.append(sentence2vec(new_sentence, diction))
    labels.append(1)  # 负标签为1
    sentences.append(sentence)

# 打乱所有的数据顺序，形成数据集
# indices为所有数据下标的一个全排列
indices = np.random.permutation(len(dataset))

# 重新根据打乱的下标生成数据集dataset，标签集labels，以及对应的原始句子sentences
dataset = [dataset[i] for i in indices]
labels = [labels[i] for i in indices]
sentences = [sentences[i] for i in indices]

# 对整个数据集进行划分，分为：训练集、校准集和测试集，其中校准和测试集合的长度都是整个数据集的10分之一
test_size = len(dataset) // 10  # " / "就表示 浮点数除法，返回浮点结果;" // "表示整数除法。
train_data = dataset[2 * test_size:]
train_label = labels[2 * test_size:]

valid_data = dataset[: test_size]
valid_label = labels[: test_size]

test_data = dataset[test_size: 2 * test_size]
test_label = labels[test_size: 2 * test_size]
# %% NN
# 输入维度为词典的大小：每一段评论的词袋模型，中间有10个隐含层神经元
model = nn.Sequential(
    nn.Linear(len(diction), 10),
    nn.ReLU(),
    nn.Linear(10, 2),
    nn.LogSoftmax(dim=1)
)


def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = torch.max(predictions.data, 1)[1]  # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    # 比如三维张量，第一个是z，第二个是行第三个是列。出来的是一个两列的矩阵，第一列是纬度上的最值，第二列是维度上的位置
    rights = pred.eq(labels.data.view_as(pred)).sum()  # 将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    #  view_as(tensor)返回被视作与给定的tensor相同大小的原tensor。
    return rights, len(labels)  # 返回正确的数量和这一次一共比较了多少元素


# %%
# 损失函数为交叉熵
cost = torch.nn.NLLLoss()
# 优化算法为Adam，可以自动调节学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
records = []

# 循环10个Epoch
losses = []
for epoch in range(10):
    for i, data in enumerate(zip(train_data, train_label)):
        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # a = [1,2,3] b = ['a','b','c']  zip(a,b) = [(1,'a'), (2, 'b'), (3, 'c')]
        x, y = data  # x:data 一行的编号，是一个array, y:label 一个值

        # 需要将输入的数据进行适当的变形，主要是要多出一个batch_size的维度，也即第一个为1的维度
        x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
        # x的尺寸：batch_size=1, len_dictionary
        # 标签也要加一层外衣以变成1*1的张量
        y = torch.tensor(np.array([y]), dtype=torch.long)
        # y的尺寸：batch_size=1, 1

        # 清空梯度
        optimizer.zero_grad()
        # 模型预测
        predict = model(x)
        # 计算损失函数
        loss = cost(predict, y)
        # 将损失函数数值加入到列表中
        losses.append(loss.data.numpy())
        # 开始进行梯度反传
        loss.backward()
        # 开始对参数进行一步优化
        optimizer.step()

        # 每隔3000步，跑一下校验数据集的数据，输出临时结果
        if i % 3000 == 0:
            val_losses = []
            rights = []
            # 在所有校验数据集上实验
            for j, val in enumerate(zip(valid_data, valid_label)):
                x, y = val
                x = torch.tensor(x, requires_grad=True, dtype=torch.float).view(1, -1)
                y = torch.tensor(np.array([y]), dtype=torch.long)
                predict = model(x)
                # 调用rightness函数计算准确度
                right = rightness(predict, y)
                rights.append(right)
                loss1 = cost(predict, y)
                val_losses.append(loss1.data.numpy())

            # 将校验集合上面的平均准确度计算出来
            right_ratio = 1.0 * np.sum([i[0] for i in rights]) / np.sum([i[1] for i in rights])
            print('第{}轮，训练损失：{:.2f}, 校验损失：{:.2f}, 校验准确率: {:.2f}'.format(epoch, np.mean(losses),
                                                                        np.mean(val_losses), right_ratio))
            records.append([np.mean(losses), np.mean(val_losses), right_ratio])
# %%
# 绘制误差曲线
a = [i[0] for i in records]
b = [i[1] for i in records]
c = [i[2] for i in records]
plt.plot(a, label='Train Loss')
plt.plot(b, label='Valid Loss')
plt.plot(c, label='Valid Accuracy')
plt.xlabel('Steps')
plt.ylabel('Loss & Accuracy')
plt.legend()
plt.title('Loss')
plt.savefig('Loss_NLP.png', dpi=72)
plt.show()
# %%
# 保存、提取模型（为展示用）
# torch.save(model, 'bow.mdl')
# model = torch.load('bow.mdl')

# %%
# 在测试集上分批运行，并计算总的正确率
vals = []  # 记录准确率所用列表

# 对测试数据集进行循环
for data, target in zip(test_data, test_label):
    data, target = torch.tensor(data, dtype=torch.float).view(1, -1), torch.tensor(np.array([target]), dtype=torch.long)
    output = model(data)  # 将特征数据喂入网络，得到分类的输出
    val = rightness(output, target)  # 获得正确样本数以及总样本数
    vals.append(val)  # 记录结果

# 计算准确率
rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
right_rate = 1.0 * rights[0].data.numpy() / rights[1]

# %%解剖神经网络
# 将神经网络的架构打印出来，方便后面的访问
model_archi = model.named_parameters
print(model_archi)
print(model[0].weight)
# %%
# 绘制出第二个全链接层的权重大小
# model[2]即提取第2层，网络一共4层，第0层为线性神经元，第1层为ReLU，第2层为第二层神经原链接，第3层为logsoftmax
plt.figure(figsize=(10, 7))
for i in range(model[2].weight.size()[0]):
    # if i == 1:
    weights = model[2].weight[i].data.numpy()
    plt.plot(weights, 'o-', label=i)
plt.legend()
plt.xlabel('Neuron in Hidden Layer')
plt.ylabel('Weights')
plt.title('Hidden Layer')
plt.savefig('neuron_Hidden_NLP', dpi=72)
plt.show()

# %%
# 将第一层神经元的权重都打印出来，一条曲线表示一个隐含层神经元。横坐标为输入层神经元编号，纵坐标为权重值大小
plt.figure(figsize=(10, 7))
for i in range(model[0].weight.size()[0]):
    # if i == 1:
    weights = model[0].weight[i].data.numpy()
    plt.plot(weights, alpha=0.5, label=i)
plt.legend()
plt.xlabel('Neuron in Input Layer')
plt.ylabel('Weights')
plt.title('Input Layer')
plt.savefig('neuron_Input_NLP', dpi=72)
plt.show()
# %%
# 将第二层的各个神经元与输入层的链接权重，挑出来最大的权重和最小的权重，并考察每一个权重所对应的单词是什么，把单词打印出来
# model[0]是取出第一层的神经元

for i in range(len(model[0].weight)):
    print('\n')
    print('第{}个神经元'.format(i))
    print('max:')
    st = sorted([(w, i) for i, w in enumerate(model[0].weight[i].data.numpy())])
    for j in range(1, 20):
        word = index2word(st[-j][1], diction)
        print(word, end=' ')
    print('\nmin:')
    for j in range(20):
        word = index2word(st[j][1], diction)
        print(word, end=' ')

# %%
# 收集到在测试集中判断错误的句子
wrong_sentences = []
targets = []
j = 0
sent_indices = []
for data, target in zip(test_data, test_label):
    predictions = model(torch.tensor(data, dtype=torch.float).view(1, -1))
    pred = torch.max(predictions.data, 1)[1]
    target = torch.tensor(np.array([target]), dtype=torch.long).view_as(pred)
    rights = pred.eq(target)
    indices = np.where(rights.numpy() == 0)[0]
    for i in indices:
        wrong_sentences.append(data)
        targets.append(target[i])
        sent_indices.append(test_size + j + i)
    j += len(target)

# %%
# 逐个查看出错的句子是什么
idx = 2
print(sent_indices)
print(sentences[sent_indices[idx]])
print(targets[idx].numpy())
lst = list(np.where(wrong_sentences[idx] > 0)[0])
mm = list(map(lambda x: index2word(x, diction), lst))
print(mm)
