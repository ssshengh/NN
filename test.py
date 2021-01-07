# %%
import numpy as np
import matplotlib.pyplot as plt
import torch

# https://www.cnblogs.com/xianhan/p/9145966.html 参考为数据集的处理

# %%
theta = np.arange(0, 1, 0.01)
x = [1, 2]
x = np.array(x, dtype=float)
y = [10, 38]
y = np.array(y, dtype=float)
line_x = []
line_y = []
a = theta[2]
# %%
c = a * x + (1 - a) * y
# %%
for i in np.arange(theta.size):
    c = theta[i] * x + (1 - theta[i]) * y
    line_x.append(c[0])
    line_y.append(c[1])
k = c[0]
# %%

plt.plot(line_x[20], line_y[20], '*')
plt.plot(line_x[10], line_y[10], '+')
plt.show()
