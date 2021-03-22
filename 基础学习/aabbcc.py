# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
nationwide = np.array([6.9, 6.9, 6.7, 6.5, 6.7, 6.4, 6.2, 6.0, 6.0, 6.1, -6.8])
First_industry = np.array([3.2, 3.4, 3.6, 3.5, 3.5, 2.7, 3.3, 2.7, 3.4, 3.1, -3.2])
Second_industry = np.array([6.2, 5.9, 5.3, 5.8, 5.8, 6.1, 5.2, 5.2, 5.8, 5.7, -9.6])
Third_industry = np.array([7.8, 8.1, 8.3, 7.8, 8.0, 7.0, 7.0, 7.2, 6.6, 6.9, -5.2])
time = np.array(['2018-1', '2018-2', '2018-3', '2018-4',
                 '2018:1-4', '2019-1', '2019-2', '2019-3', '2019-4', '2019:1-4', '2020-1'])
# %%
plt.plot(time, nationwide, 'bo', time, nationwide, 'b')
plt.show()
# %%
fig = plt.figure(figsize=(19, 8), dpi=360)
ax1 = fig.add_subplot(111)
ga = plt.gca()

ax1.set(xlim=[-0.3, 11], ylim=[-12, 12], title='China GDP Growth Rate (2018-2019)',
        ylabel='Growth Rate', xlabel='Quarter')
lines = plt.plot(time, nationwide, '^-', time, First_industry, '*--',
                 time, Second_industry, 'p--', time, Third_industry, 'o--')
plt.setp(lines[0], linewidth=1, markersize=10)
plt.setp(lines[1], linewidth=1, markersize=10)
plt.setp(lines[2], linewidth=1, markersize=10)
plt.setp(lines[3], linewidth=1, markersize=10)

# 把每个值添加到点上
for i in range(len(nationwide)):
    plt.text(time[i], nationwide[i] + 0.5, '%s' % round(nationwide[i], 3), va='baseline', ha='right', fontsize=10)
    plt.text(time[i], First_industry[i] - 1, '%s' % round(First_industry[i], 3), va='center', ha='right', fontsize=10)
    plt.text(time[i], Second_industry[i] - 1, '%s' % round(Second_industry[i], 3),
             ha='center', fontsize=10, va='bottom')
    plt.text(time[i], Third_industry[i] + 0.5, '%s' % round(Third_industry[i], 3), va='bottom', ha='left', fontsize=10)
plt.legend(('Nationwide', 'Primary Industry', 'Second Industry', 'Third Industry'),
           loc='best')
# plt.title('GDP Growth in China')
plt.savefig('first.png', dpi=2080)
plt.show()
# %%
label = 'GDP', 'Primary Industry', 'Second Industry', 'Third Industry'
data_sum_all = [206504, 10186, 73638, 122680]
data_rate_all = [-6.8, -3.2, -9.6, -5.2]
label_more = ['AFAHF', 'manufacturing', 'building', 'retail', 'postal', 'accommodation catering', 'financial', 'realty',
              'information', 'services', 'lease', 'other services']
data_sum_more = [10708, 64642, 53852, 9378, 18750, 7865, 2821, 21347, 15268, 8928, 7138, 39660]
data_rate_more = [-2.8, -8.5, -10.2, -17.5, -17.8, -14, -35.3, 6, -6.1, 13.2, -9.4, -1.8]


# %%
# 冒泡
def bubble_sort(arr, k):
    if arr is None and len(arr) < 2:
        return
    for end in range(len(arr) - 1, -1, -1):
        for i in range(end):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                k[i], k[i + 1] = k[i + 1], k[i]
    return arr, k


y_data, x_data = bubble_sort(data_rate_more, label_more)

# %%
# fig = plt.figure(figsize=(19, 8), dpi=180)
plt.subplot(221)

# 绘制饼图

# 这些都没必要
# # 将横、纵坐标轴标准化处理,保证饼图是一个正圆,否则为椭圆
# plt.axes(aspect='equal')
# # # 控制X轴和Y轴的范围(用于控制饼图的圆心、半径)
# # plt.xlim(-5.2, -2)
# # plt.ylim(-0.2, 3)
# # 不显示边框
# ga = plt.gca()
# ga.spines['top'].set_color('none')
# ga.spines['right'].set_color('none')
# ga.spines['left'].set_color('none')
# ga.spines['bottom'].set_color('none')
# # 不显示X轴、Y轴的刻度值
# plt.xticks(())
# plt.yticks(())

# explode = [0.05, 0.08, 0.08]
# colors = ['red', 'pink', 'orange']
# plt.pie(x=data_sum_all[1:],  # 绘制数据
#         labels=label[1:],  # 添加编程语言标签
#         explode=explode,  # 突出显示
#         colors=colors,  # 设置自定义填充色
#         autopct='%.3f%%',  # 设置百分比的格式,保留3位小数
#         pctdistance=0.5,  # 设置百分比标签和圆心的距离
#         labeldistance=1.1,  # 设置标签和圆心的距离
#         startangle=180,  # 设置饼图的初始角度
#         center=(1, 0.4),  # 设置饼图的圆心(相当于X轴和Y轴的范围)
#         radius=0.3,  # 设置饼图的半径(相当于X轴和Y轴的范围)
#         counterclock=False,  # 是否为逆时针方向,False表示顺时针方向
#         wedgeprops={'linewidth': 1, 'edgecolor': 'green'},  # 设置饼图内外边界的属性值
#         textprops={'fontsize': 20, 'color': 'black'},  # 设置文本标签的属性值
#         frame=1)  # 是否显示饼图的圆圈,1为显示
plt.pie(x=data_sum_all[1:], labels=label[1:], autopct='%.3f%%', explode=[0, 0.05, 0])
# 添加图形标题
plt.title('Distribution of China\'s total GDP')

plt.subplot(222)
ga = plt.gca()
ga.spines['top'].set_color('none')
ga.spines['right'].set_color('none')
ga.spines['left'].set_color('none')
ga.spines['bottom'].set_color('none')
plt.show()
# %%
fig = plt.figure(figsize=(19, 8), dpi=180)
plt.subplot(221)
plt.pie(x=data_sum_all[1:],
        labels=label[1:],
        autopct='%.3f%%',
        explode=[0, 0.05, 0])
plt.title('Distribution of China\'s total GDP')
plt.legend(loc="best", fontsize=10, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.3)
# loc =  'upper right' 位于右上角
# bbox_to_anchor=[0.5, 0.5] # 外边距 上边 右边
# ncol=2 分两列
# borderaxespad = 0.3图例的内边距

plt.subplot(222)
explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
plt.pie(x=data_sum_more,
        labels=label_more,
        explode=explode,  # 突出显示
        autopct='%.3f%%')
plt.title('Subdivide of China\'s total GDP')
plt.legend(loc="best", fontsize=5, bbox_to_anchor=(1.1, 1.05), borderaxespad=0.3)

plt.subplot(212)
"""
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data', 0))
"""
plt.ylim(-40, 20)
plt.xlabel('Different Industries')
plt.ylabel('Growth Rate (%)')
plt.title('The Growth Rate in China\'s Different Industries')
plt.xticks(rotation=10)  # 横坐标斜过来
for i in range(len(data_sum_more)):
    if data_rate_more[i] > 0:
        plt.bar(label_more[i], data_rate_more[i], facecolor='#9999ff', edgecolor='white')
        plt.text(label_more[i], data_rate_more[i], '%s' % round(data_rate_more[i], 3), va='bottom', ha='left')
    else:
        plt.bar(label_more[i], data_rate_more[i], facecolor='#ff9999', edgecolor='white')
        plt.text(label_more[i], data_rate_more[i], '%s' % round(data_rate_more[i], 3), va='top', ha='left')

plt.text(label_more[8], data_rate_more[11] - 30, 'NOTE: The Total GDP of China is 206504 Hundred Million',
         fontdict={'size': 10, 'color': 'pink'})

plt.savefig('second.png', dpi=256)
plt.show()
# %%
x_data = ['Poultry Meat', 'Manufacturing Sector', 'Construction Sector', 'Transportation Sector',
          'Hotel and Catering Sector', 'Wholesale and Retail Sector', 'Real Estate Sector', 'Financial Sector']
y_data = [-19.5, -10.2, -17.5, -14.0, -35.3, -17.8, -6.1, 6.0]
y_data, x_data = bubble_sort(y_data, x_data)

# %%
import mpl_toolkits.axisartist as axisartist

# 创建画布
fig = plt.figure(figsize=(19, 8), dpi=180)
# 使用axisartist.Subplot方法创建一个绘图区对象ax
ax = axisartist.Subplot(fig, 111)
# 将绘图区对象添加到画布中
fig.add_axes(ax)

ax.axis[:].set_visible(False)  # 通过set_visible方法设置绘图区所有坐标轴隐藏
ax.axis["x"] = ax.new_floating_axis(0, 0)  # ax.new_floating_axis代表添加新的坐标轴
ax.axis["x"].set_axisline_style("->", size=1.0)  # 给x坐标轴加上箭头
"""
# 添加y坐标轴，且加上箭头
ax.axis["y"] = ax.new_floating_axis(1, 0)
ax.axis["y"].set_axisline_style("-|>", size=1.0)
"""
# 设置x、y轴上刻度显示方向
ax.axis["x"].set_axis_direction("top")
# ax.axis["y"].set_axis_direction("right")

plt.ylim(-40, 20)
# plt.xlabel('Different Sectors')
plt.ylabel('Growth Rate (%)')
plt.title('The Growth Rate of Specific Sectors (2020 Q1)')
plt.xticks(rotation=10)  # 横坐标斜过来

# 显示柱形图不同柱形的颜色
for i in range(len(y_data)):
    if y_data[i] > 0:
        plt.bar(x_data[i], y_data[i], facecolor='#9999ff', edgecolor='white')
        plt.text(x_data[i], y_data[i], '%s' % round(y_data[i], 3), va='bottom', ha='left')
    else:
        plt.bar(x_data[i], y_data[i], facecolor='#ff9999', edgecolor='white')
        plt.text(x_data[i], y_data[i], '%s' % round(y_data[i], 3), va='top', ha='left')
# plt.grid(True)
# plt.text(x_data[5], y_data[7] - 40, 'NOTE: The Total GDP of China is 206504 Hundred Million',
#          fontdict={'size': 10, 'color': 'pink'})
plt.yticks([])
plt.annotate('Different Sectors', (x_data[7], 0), xycoords='data', xytext=(x_data[7], -10), fontsize=15)
plt.savefig('second.png', dpi=2080)
plt.show()

# %%
fig = plt.figure(figsize=(19, 8), dpi=180)

"""
plt.xticks([])  #去掉x轴
plt.yticks([])  #去掉y轴
plt.axis('off')  #去掉坐标轴

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)
"""
fig_size = (19, 8)
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_position(('data', 0))
ax.spines['left'].set_color('none')
ax.arrow(0, (1 - 0.04 * fig_size[0] / fig_size[1]) * y_max, 0, 0.04 * fig_size[0] / fig_size[1] * y_max,
         head_width=0.04 * fig_size[1] / fig_size[0] * x_max,
         head_length=0.04 * fig_size[0] / fig_size[1] * y_max,
         fc='black',
         length_includes_head=True)

plt.ylim(-40, 20)
# plt.xlabel('Different Sectors')
plt.ylabel('Growth Rate (%)')
plt.title('The Growth Rate of Specific Sectors (2020 Q1)')
plt.xticks(rotation=10)  # 横坐标斜过来
for i in range(len(y_data)):
    if y_data[i] > 0:
        plt.bar(x_data[i], y_data[i], facecolor='#9999ff', edgecolor='white')
        plt.text(x_data[i], y_data[i], '%s' % round(y_data[i], 3), va='bottom', ha='left')
    else:
        plt.bar(x_data[i], y_data[i], facecolor='#ff9999', edgecolor='white')
        plt.text(x_data[i], y_data[i], '%s' % round(y_data[i], 3), va='top', ha='left')
# plt.grid(True)
# plt.text(x_data[5], y_data[7] - 40, 'NOTE: The Total GDP of China is 206504 Hundred Million',
#          fontdict={'size': 10, 'color': 'pink'})
plt.yticks([])
plt.annotate('Different Sectors', (x_data[7], 0), xycoords='data', xytext=(x_data[7], -10), fontsize=15)
plt.savefig('second.png', dpi=256)
plt.show()
# %%

