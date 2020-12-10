# %%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import os
import math
from torch.utils.data.dataloader import DataLoader

#%%
BATCH_SIZE = 100
TRAIN_EPOCHS = 10
CLASSES = 10
INPUT_HEIGHT = 28
INPUT_WIDTH = 28
TOTAL_INPUT = 784
TRAIN_SIZE = 50000
TEST_SIZE = 10000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%%
tr = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, ), (0.5, ))
])
minst = torchvision.datasets.mnist.MNIST(
    root='./data',
    transform=tr,
    download=False
)
train_set, test_set = torch.utils.data.random_split(minst, lengths=(TRAIN_SIZE, TEST_SIZE))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
#%%
SAMPLES = 10
PI = 0.5
SIGMA1 = torch.FloatTensor([math.exp(-0)])
SIGMA2 = torch.FloatTensor([math.exp(-6)])


# %%
class Gaussian:
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0, 1)

    @property
    def sigma(self):
        # 计算 input+1的自然对数 yi=log(xi+1) 注意：对值比较小的输入，此函数比torch.log()更准确。
        return torch.log1p(torch.exp(self.rho))

    # 采样后反归一化
    def sample(self):
        # 如果分配参数是批处理的，则生成sample_shape形状的样本或sample_shape形状的样本批。
        epsilon = self.normal.sample(self.mu.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, input):
        return (
                -math.log(math.sqrt(2 * math.pi)) - torch.log(self.sigma) - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)
        ).sum()


# %%
class ScaledGaussianMixture:
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, input):
        # log_prob(value),返回以(value)评估的概率密度/质量函数的对数。可能只是想要把input w给放入正太分布？
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1 - self.pi) * prob2)).sum(), prob1, prob2


pi1 = math.pi
sigma11 = 1.1
sigma21 = 0.01
input1 = torch.from_numpy(np.arange(0, 100, 0.5))

sGM = ScaledGaussianMixture(pi1, sigma11, sigma21)
[mixFun, p1, p2] = sGM.log_prob(input1)
p1 = p1.numpy()
# plt.plot(p1)
plt.plot(p2)
plt.plot(mixFun)
plt.show()

# %%
dim_out = 2
dim_in = 5
a = (-0.2 - 0.2) * torch.rand(dim_out, dim_in)
b = a + 0.2
c = -5. + 4.
w_mu = nn.Parameter((-0.2 - 0.2) * torch.rand(dim_out) + 0.2)
"""
nn.Parameter()
Variable的一种，常被用于模块参数(module parameter)。
Parameters 是 Variable 的子类。Paramenters和Modules一起使用的时候会有一些特殊的属性，
即：当Paramenters赋值给Module的属性的时候，他会自动的被加到 Module的 参数列表中(即：会出现在 parameters() 迭代器中)。
将Varibale赋值给Module属性则不会有这样的影响。 这样做的原因是：我们有时候会需要缓存一些临时的状态(state), 
比如：模型中RNN的最后一个隐状态。如果没有Parameter这个类的话，那么这些临时变量也会注册成为模型变量。
Variable 与 Parameter的另一个不同之处在于，Parameter不能被 volatile(即：无法设置volatile=True)而且默认requires_grad=True。Variable默认requires_grad=False。
"""


# %%
class BayesianLinear(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(BayesianLinear, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        # 因为我们需要假定我们的高斯分布的参数，需要mean & std ，但是使用rho更为有效一些（它是sigma的变形（std））
        self.w_mu = nn.Parameter((-0.2 - 0.2) * torch.rand(dim_out, dim_in) + 0.2)
        self.w_rho = nn.Parameter((-5. + 4.) * torch.rand(dim_out, dim_in) - 4.0)
        self.w = Gaussian(self.w_mu, self.w_rho)

        self.b_mu = nn.Parameter((-0.2 - 0.2) * torch.rand(dim_out) + 0.2)
        self.b_rho = nn.Parameter((-5. + 4.) * torch.rand(dim_out) - 4.0)
        self.b = Gaussian(self.b_mu, self.b_rho)

        self.w_prior = ScaledGaussianMixture(PI, SIGMA1, SIGMA2)
        self.b_prior = ScaledGaussianMixture(PI, SIGMA1, SIGMA2)
        self.log_prior = 0
        self.log_variational_post = 0

    def forward(self, input, sample = False, calc_log_rob=False):
        if self.training or sample:
            w = self.w.sample()
            b = self.b.sample()
        else:
            w = self.w.mu
            b = self.b.mu

        if self.training or calc_log_rob:
            self.log_prior = self.w_prior.log_prob(w) + self.b_prior.log_prob(b)
            self.log_variational_post = self.w.log_prob(w) + self.b.log_prob(b)
        else:
            self.log_prior, self.log_variational_post = 0, 0

        return F.linear(input, w, b)