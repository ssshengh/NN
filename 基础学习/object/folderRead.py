# %%
import os
import re


# %%
def get_file_names(path: str):
    """
    从给定路径中读取json文件路径
    按照攻击车和普通车，分别将路径存入四个list内
    :param path: 存放json文件的路径
    :return: 四个list，一个存储攻击车路径，一个存储普通车路径，另外两个分别是这两种车的编号
    """
    # 两个匹配两种车的正则表达
    attacker = re.compile("(A1)")
    common = re.compile("(A0)")

    # 分别存储两种车的路径的数组
    list_attack = []
    list_common = []

    # 分别储存两种车读取到的编号
    id_attack = []
    id_common = []

    count = 0  # 统计文件数量
    for i in os.listdir(path):
        a = attacker.search(i)
        b = common.search(i)
        if a:
            print("攻击车：" + i)
            id1 = get_id(i)
            id_attack.append(id1)
            list_attack.append(path + i)
            count += 1
        elif b:
            print("receiver: " + i)
            id2 = get_id(i)
            id_common.append(id2)
            list_common.append(path + i)
            count += 1
    print("文件数： " + str(count))

    return list_attack, id_attack, list_common, id_common


# %%

def get_id(string: str) -> int:
    get_ID = re.compile("\d+")
    it = get_ID.findall(string)
    ID = int(it[1])
    print("该车编号为："+str(ID)+'\n')
    return ID
