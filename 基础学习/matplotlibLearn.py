# 导入 matplotlib 的所有内容（nympy 可以用 np 这个名字来使用）
from pylab import *
import numpy
import matplotlib.pyplot as plt

X = numpy.linspace(-np.pi, np.pi, 256, endpoint=True)
# print(X)
c, s = numpy.cos(X), numpy.sin(X)

plt.plot(X, c)
plt.plot(X, s)

# plt.show()
print("***************************************************************************************")

# 创建一个 8 * 6 点（point）的图，并设置分辨率为 80
figure(figsize=(8, 6), dpi=80)

# 创建一个新的 1 * 1 的子图，接下来的图样绘制在其中的第 1 块（也是唯一的一块）
subplot(2, 1, 1)

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

# 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
plot(X, C, color="blue", linewidth=1.0, linestyle="-")

# 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
plot(X, S, color="green", linewidth=1.0, linestyle="-")

# 设置横轴的上下限
xlim(-4.0, 4.0)

# 设置横轴记号
xticks(np.linspace(-4, 4, 9, endpoint=True))

# 设置纵轴的上下限
ylim(-1.0, 1.0)

# 设置纵轴记号
yticks(np.linspace(-1, 1, 5, endpoint=True))

# 移动脊柱
ax = gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))

# 设置边界
xmin, xmax = X.min(), X.max()
ymin, ymax = C.min(), C.max()

dx = (xmax - xmin) * 0.2
dy = (ymax - ymin) * 0.2

xlim(xmin - dx, xmax + dx)
ylim(ymin - dy, ymax + dy)

# 设置记号
xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
yticks([-1, 0, +1])

# 设置记号的标签
xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
       [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

yticks([-1, 0, +1],
       [r'$-1$', r'$0$', r'$+1$'])

# 添加图例
plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine")

legend(loc='upper left')

# 给一些特殊点做注释
t = 2 * np.pi / 3
plot([t, t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
scatter([t, ], [np.cos(t), ], 50, color='blue')

annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
         xy=(t, np.sin(t)), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plot([t, t], [0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
scatter([t, ], [np.sin(t), ], 50, color='red')

annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
         xy=(t, np.cos(t)), xycoords='data',
         xytext=(-90, -50), textcoords='offset points', fontsize=16,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

# 以分辨率 72 来保存图片
# savefig("exercice_2.png",dpi=72)
# 在屏幕上显示
show()

print("**********************************************************************************")
