# -*- coding: utf-8 -*-
"""
Created on Mon May  8 20:38:08 2017

@author: xiao
"""

K = [0,8]

train_y = []
# 提取训练数据需要的数据集
for i in range(0,len(K)):
    train_y.extend(Train_y[Train_y == K[i]])
train_y = np.array(train_y)
train_x = []
for i in range(0,len(K)):
    train_x.extend(list(Train_x[Train_y == K[i],:]))
train_x = np.array(train_x)
# 提取测试数据需要的数据集
test_m,test_n = Test_x.shape
test_y = []
for i in range(0,len(K)):
    test_y.extend(Test_y[Test_y == K[i]])
test_y = np.array(test_y)
test_x = []
for i in range(0,len(K)):
    test_x.extend(list(Test_x[Test_y == K[i],:]))
test_x = np.array(test_x)
# 将数据集分成2类
train_x1 = train_x[train_y == K[0],:]
train_x2 = train_x[train_y == K[1],:]

W,w0 = train_FDA(train_x1,train_x2)
pre_y = test_FDA(test_x,w0,W,K)
pre_y = pre_y.reshape((len(pre_y)))
pre_y = OFK(pre_y,K)
test_y = OFK(test_y,K) # 将测试集的标签转化为矩阵形式
accuary = 1 - sum(sum(abs(pre_y - test_y)))/(2*test_m) # 计算正确率
print('accuary = ',accuary)
