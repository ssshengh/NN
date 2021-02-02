# %%
import pandas as pd

conformed = pd.read_csv('/Users/shenheng/Downloads/covid192815/time_series_covid19_confirmed_global.csv')
deaths = pd.read_csv('/Users/shenheng/Downloads/covid192815/time_series_covid19_deaths_global.csv')
recovered = pd.read_csv('/Users/shenheng/Downloads/covid192815/time_series_covid19_recovered_global.csv')
# %%
conformed = conformed[58:91]
deaths = deaths[58:91]
recovered = recovered[43:76]
# %%
print(conformed)
# %%
# 计算各列数据总和并作为新列添加到末尾 df['Col_sum'] = df.apply(lambda x: x.sum(), axis=1)
# 计算各行数据总和并作为新行添加到末尾 df.loc['Row_sum'] = df.apply(lambda x: x.sum())
import numpy as np
import matplotlib.pyplot as plt


# %%
def getData(dataframe):
    dataframe.loc['row_sum'] = dataframe.apply(lambda x: x.sum())
    # 删除前三列
    dataframe = dataframe.drop(["Province/State", "Country/Region", "Lat", "Long"], axis=1)
    # 只要最后一行
    dataframe = dataframe[-1:]
    data = dataframe.values
    # 获取列名称
    label = dataframe.columns
    label = label.tolist()
    label = np.array(label)
    return data, label


# %%
data_conformed, label_conformed = getData(conformed)
data_death, label_death = getData(deaths)
data_recovered, label_recovered = getData(recovered)

# %%
data_death = np.reshape(data_death, (373,))
data_recovered = np.reshape(data_recovered, (373,))
data_conformed = np.reshape(data_conformed, (373,))
#%%
import matplotlib.ticker as ticker
# %%
fig = plt.figure(figsize=(19, 8), dpi=360)
plt.xticks(rotation=45)  # 横坐标斜过来
# MultipleLocator里面的值是间隔多少个显示
plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(20))
plt.ylabel('Nums of Human')
plt.grid(True)
plt.plot(label_death, data_death, 'r', label_recovered, data_recovered, 'g--', label_conformed, data_conformed, 'b-.')
plt.annotate('Date', (label_death[-1], 0), xycoords='data', xytext=(label_death[-1], -20000), fontsize=15)
plt.legend(('Deaths', 'Recovered', 'Conformed'), loc='best')
plt.savefig('Third.png', dpi=2080)
plt.show()
