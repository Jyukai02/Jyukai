import os
import csv
import numpy as np

path = 'C:\\Users\\46562\\Desktop\\Rnn datasets\\' # 存储csv的位置
dirs = os.listdir(path)  # 返回指定的文件夹包含的文件的名字的列表
for x in dirs:  # 查找列表中的csv文件
    if os.path.splitext(x)[1] == 'project3_train.csv':
        filePath = x
        break
with open(x, 'r') as f:  # 读取csv文件

    data = csv.reader(f)
    for i in data:
        file = open('C:\\Users\\46562\\Desktop\\Rnn datasets\\test1\\' + str(i[0]) + '-' + str(i[1]) + '.c', 'wb+')  # 打开文件
        file.write(str(i[0]).encode(encoding='utf-8'))  # 写入文件
        file.close()
    print("done")