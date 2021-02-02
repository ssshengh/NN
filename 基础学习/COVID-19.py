# %%
import requests
import json
import csv


#
# parse_text1.py
def parse_txt(url):
    headers = {
        'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36 QIHU 360EE'
    }
    response = requests.get(url, headers=headers)  # 发起请求
    words = json.loads(response.content)  # json字符串转换为Python类型
    # 响应对象保存为json格式文件
    with open("/Users/shenheng/Downloads/CO/feiyan_new.json", "w",
              encoding='utf-8') as f:
        f.write(json.dumps(words, indent=2, ensure_ascii=False))
        print("保存成功！")


if __name__ == "__main__":
    url = "https://view.inews.qq.com/g2/getOnsInfo?name=disease_h5"
    parse_txt(url)

#%%
fp = open("/Users/shenheng/Downloads/CO/feiyan_new.json", "r", encoding='utf-8')
fp_read = fp.read()
st = json.loads(fp_read)
sta = json.loads(st['data'])

# 我国疫情总体情况
chinaTotals = "确诊人数: "+str(sta['chinaTotal']['confirm'])+" 疑似人数: " +\
    str(sta['chinaTotal']['suspect']) + " 死亡人数:" +\
    str(sta['chinaTotal']['dead'])+" 治愈人数: "+str(sta['chinaTotal']['heal']) +\
    " 更新日期:"+sta['lastUpdateTime']
print(chinaTotals)

# 获取中国各省名称，确诊人数，疑似人数，死亡人数，治愈人数
# 从爬取的信息中提取所需信息
china = sta['areaTree'][0]['children']
csvfile = open("/Users/shenheng/Downloads/CO/newIlness.csv", 'w', encoding='utf-8', newline='')

for i in range(len(china)):
    writer = csv.writer(csvfile)
    writer.writerow([china[i]['name']])
    # 打印目前为止已知的确诊人数
    print(china[i]['name'], '确诊:' + str(china[i]['total']['nowConfirm']), '死亡:' +
          str(china[i]['total']['dead']), '治愈:' + str(china[i]['total']['heal']))
    for city in china[i]['children']:
        # 写入市的名称，确诊、死亡、治愈的人数
        writer = csv.writer(csvfile)
        writer.writerow([city['name'], '确诊:' + str(city['total']['confirm']), '死亡:' +
                         str(city['total']['dead']), '治愈:' + str(city['total']['heal'])])
print("保存成功！")
fp.close()