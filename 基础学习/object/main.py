import OneFile as F
import folderRead as fR

if __name__ == '__main__':
    # path = '/Users/shenheng/Downloads/cO/securecomm2018/results/JSONlog-34-211-A0.json'
    # path_after = '/Users/shenheng/Code/NN/基础学习/object/data/211.json'
    # oneFile = F.OneFileHandle(path, path_after)
    # data_original, type = oneFile.read_data()
    # oneFile.write2txt('/Users/shenheng/Code/NN/bb.csv')

    path_json = '/Users/shenheng/Downloads/CO/securecomm2018/results/'
    list_attack, id_a, list_common, id_c = fR.get_file_names(path_json)
    print("read files, yes!")

    path_after = '/Users/shenheng/Code/NN/基础学习/object/data/'
    path_write = '/Users/shenheng/Code/NN/data/'

    print(id_c)

    for p in range(len(list_attack)):
        # 初始化处理单个文件对象进行车辆处理
        path_to = path_after+str(id_a[p])+'-attack.json'
        print("修正json路径："+path_to)
        path_w = path_write + str(id_a[p]) + '-attack.csv'
        onefile = F.OneFileHandle(list_attack[p], path_to)
        onefile.read_data()
        onefile.write2txt(path_w)

    for p in range(len(list_common)):
        # 初始化处理单个文件对象进行车辆处理
        path_to = path_after+str(id_c[p])+'-common.json'
        print("修正json路径："+path_to)
        path_w = path_write + str(id_c[p]) + '-common.csv'
        onefile = F.OneFileHandle(list_common[p], path_to)
        onefile.read_data()
        onefile.write2txt(path_w)

