import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os


class OneFileHandle:

    def __init__(self, path_from: str, path_to: str):
        """
        构造器，需要传入文件的路径，初始化文件路径以及新生成的可读的json文件路径
        """
        if path_from.find(".json"):
            print("输入文件为json文件")
            self.path_original = path_from  # 原始json文件位置
            self.path_destination = path_to  # 新建json文件位置

            print("在给出目录下新建json文件以存储可处理数据文件")
            self.__create_file(self.path_destination)

            self.data_original = pd.DataFrame  # 存储读取的原始数据
            self.data = pd.DataFrame  # 存储删除sender为nan后的数据
            self.sender = []  # 列表储存该文件里的发送者有哪些

        else:
            print("输入文件格式不是json，请检查")

    def read_data(self):
        """
        私有方法
        处理数据，将json处理为可以读取的格式，然后读取存入成为一个Dataframe
        :return: 返回读取的原始数据以及数据中的发送车辆的编号
        """

        self.__get_json(self.path_original, self.path_destination)  # 完成数据处理，添加逗号及中括号，存为新的json文件
        self.data_original = pd.read_json(self.path_destination)  # 读取数据为dataFrame
        self.data = self.data_original.drop_duplicates(subset=['sender'], keep='first')  # first的意思是保留重复数据的一个
        type = self.data['sender'].tolist()
        print("本文件里的发送者车辆编号为：\n" + str(type))
        return self.data_original, type

    def write2txt(self, path2write: str):
        """
        这个函数的功能是，从一个接收车文件中得到一张dataFrame数据，将该数据进行处理：
        按照每两个数据位置及速度相减得到一个数据，将该数据写入一个csv文件中
        :param df: 输入的dataFrame数据表
        :param path2write: 输出数据的位置及表的名称
        :return: 写入的数据
        """
        self.__create_file(path2write)

        # 首先丢弃sender为nan的数据：
        index_drop = list(self.data_original[np.isnan(self.data_original['sender'])].index)
        self.data = self.data_original.drop(index_drop)  # 丢弃了为nan的数据存入data中
        # 再把所有的sender种类存入数组
        df_list = self.data['sender'].unique()

        for senderI in range(len(df_list)):
            sen = self.data[self.data['sender'].isin([df_list[senderI]])]  # 逐个取出其中的
            self.PV2txt(sen, path2write, df_list[senderI])

    def PV2txt(self, df: pd.DataFrame, path2write: str, ID):
        """
        这个函数完成的任务是：从一个接收车文件表中提取出来的某个发车的子dataframe表df
        将df中的位置以及速度信息两两相减存入数组，并写入txt文件
        :param attack: bool变量，判别是否是攻击车
        :param ID: ID是发送车编号
        :param path2write: 输出数据的位置及表的名称
        :param df: 输入的dataFrame数据表的某个发车的子表
        :return:
        """
        attack = self.__attack_id()
        print("写入第{}车的数据".format(ID))
        # 用两个列表来分别存储位置信息与速度信息，方便处理
        list_write_pos = []
        list_write_spd = []
        # 接下来从子表中分别提取位置数据与速度数据存入列表
        for index, row in df.iterrows():
            list_write_pos.append(row['pos'])
            list_write_spd.append(row['spd'])
        # 开始写入文件,a+意思是在后面接着写入
        with open(path2write, 'a+') as f:
            for i in range(len(list_write_spd) - 1):
                # list需要换位set减之后换回list方便处理，write写入的是string，所以还需要转换为str
                pos = self.__listSubtract(list_write_pos[i + 1], list_write_pos[i])
                spd = self.__listSubtract(list_write_spd[i + 1], list_write_spd[i])
                if ID in attack:
                    f.write(str(pos[0]) + ',' + str(pos[1])
                            + ','
                            + str(spd[0]) + ',' + str(spd[1])
                            + ', 1, '
                            + str(ID) + '\n')
                else:
                    f.write(str(pos[0]) + ',' + str(pos[1])
                            + ','
                            + str(spd[0]) + ',' + str(spd[1])
                            + ', 0, '
                            + str(ID) + '\n')
            # f.write('[' + list_write[i] + ',' + list_write[i+1]  + ', 0' + ']' + '\n') #表示是未攻击车辆

    # 创建文件方法：
    @staticmethod
    def __create_file(filename):
        """
        创建日志文件夹和日志文件
        为静态文件，可以单独调出来创建文件
        :param filename:文件的绝对路径
        :return:如果不存在文件夹或者文件，新建的文件和文件夹
        """
        path = filename[0:filename.rfind("/")]
        # Python rfind() 返回字符串最后一次出现的位置，如果没有匹配项则返回 -1
        if not os.path.isdir(path):  # 无文件夹时创建
            print("没有该文件夹{}，创建文件夹".format(path))
            os.makedirs(path)
        if not os.path.isfile(filename):  # 无文件时创建
            print("没有该文件，创建文件")
            fd = open(filename, mode="w", encoding="utf-8")
            fd.close()
        else:
            print("该文件{}已经存在".format(filename))
            pass

    # 处理json文件，添加逗号及中括号，用以读;然后得到读取的json文件
    def __get_json(self, path, path_after):
        with open(path, 'r') as f1, open(path_after, 'w') as f2:
            line = f1.readlines()  # 读出为一个序列
            line[0] = '[' + line[0]
            for i in range(len(line) - 1):
                # line_new =line_list.replace('\n','')# 将换行符替换为空('')
                line[i] = line[i] + ','
                f2.writelines(line[i])
            line[-1] = line[-1] + ']'
            f2.writelines(line[-1])

    # 一个简单的用于list相减的方法
    def __listSubtract(self, list1: list, list2: list) -> list:
        """
        将两个list相减：[a1,b1]-[a2,b2],list1-lis
        :param list1:
        :param list2:
        :return: 得到的list
        """
        res = list(map(lambda x: x[0] - x[1], zip(list2, list1)))
        return res

    def __attack_id(self)->list:
        ID = [79, 205, 49]
        return ID
