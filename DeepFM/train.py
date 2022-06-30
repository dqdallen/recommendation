'''
Author: dqdallen 1986360994@qq.com
Date: 2022-06-14 21:03:28
LastEditors: dqdallen 1986360994@qq.com
LastEditTime: 2022-06-15 14:55:33
FilePath: \recommendation\DeepFM\train.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import random
from sklearn import preprocessing
import pandas as pd


f = open('./DeepFM_with_PyTorch/data/raw/train.txt', 'r')
data = pd.read_csv('./DeepFM_with_PyTorch/data/raw/train.txt',
                   nrows=1000000, sep='\t', header=None)
data = data.fillna(value='-99')

data.columns = [str(i) for i in range(0, 40)]
data[[str(i) for i in range(1, 14)]] = data[[str(i)
                                             for i in range(1, 14)]].astype('float64')
data1 = data[data['0'] == 1]
data2 = data[data['0'] == 0].iloc[0:254949, :]
data = pd.concat([data1, data2])

dic = {}
for i in range(14, 40):
    dic[i] = set()
for i in range(14, 40):
    tmp = set(data.iloc[:, i])
    dic[i] = dic[i] | tmp

for i in range(14, 40):
    enc = preprocessing.LabelEncoder()
    enc.fit(list(dic[i]))
    tmp = enc.transform(data.loc[:, str(i)])
    data[str(i)] = tmp
fea_nums_dict = {}
for i in range(14, 40):
    fea_nums_dict[str(i)] = len(dic[i])

data1_train = data.iloc[:254949, :].iloc[:250000, :]
data1_test = data.iloc[:254949, :].iloc[250000:, :]
data2_train = data.iloc[254949:, :].iloc[:250000, :]
data2_test = data.iloc[254949:, :].iloc[250000:, :]
data_train = pd.concat([data1_train, data2_train])
data_test = pd.concat([data1_test, data2_test])

idx = list(range(data_train.shape[0]))
random.shuffle(idx)
data_train = data_train.iloc[idx, :]
idx = list(range(data_test.shape[0]))
random.shuffle(idx)
data_test = data_test.iloc[idx, :]
