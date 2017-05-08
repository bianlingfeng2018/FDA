# -*- coding: utf-8 -*-
"""
Created on Mon May  8 19:31:04 2017

@author: xiao
"""
import os
import numpy as np
import struct
# Fisher线性判别分析
os.chdir(r"C:\Users\xiao\.spyder-py3\机器学习")
def loadImageSet(filename):  
  
    binfile = open(filename, 'rb') # 读取二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>IIII', buffers, 0) # 取前4个整数，返回一个元组,I表示int数据  
  
    offset = struct.calcsize('>IIII')  # 定位到data开始的位置  
    imgNum = head[1]  
    width = head[2]  
    height = head[3]  
  
    bits = imgNum * width * height  # data一共有60000*28*28个像素值  
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'  
  
    imgs = struct.unpack_from(bitsString, buffers, offset) # 取data数据，返回一个元组  
  
    binfile.close()  
    imgs = np.reshape(imgs, [imgNum, width * height]) # reshape为[60000,784]型数组  
  
    return imgs  
  
  
def loadLabelSet(filename):  
  
    binfile = open(filename, 'rb') # 读二进制文件  
    buffers = binfile.read()  
  
    head = struct.unpack_from('>II', buffers, 0) # 取label文件前2个整形数  
  
    labelNum = head[1]  
    offset = struct.calcsize('>II')  # 定位到label数据开始的位置  
  
    numString = '>' + str(labelNum) + "B" # fmt格式：'>60000B'  
    labels = struct.unpack_from(numString, buffers, offset) # 取label数据  
  
    binfile.close()  
    labels = np.reshape(labels, [labelNum]) # 转型为列表(一维数组)  
  
    return labels
# 进行使用测试集去计算手写数字标签    
def test_FDA(x,w0,W,K):
    Y = np.dot(x,W)
    pre_y = np.zeros((test_x.shape[0],1))
    for i in range(0,test_x.shape[0]):
        if Y[i] > w0:
            pre_y[i] = K[0]
        else:
            pre_y[i] = K[1]
    return pre_y
    
def train_FDA(x1,x2):
    n= x1.shape[1]
    m1 = np.mean(x1,axis = 0).reshape(1,n)
    m2 = np.mean(x2,axis = 0).reshape(1,n)
    S1 = np.dot((x1 - m1).T,(x1 - m1)) # 在这里S1的维度为 784*784
    S2 = np.dot((x2 - m2).T,(x2 - m2))
    SW = S1 + S2
    W = np.dot(np.linalg.pinv(SW),(m1-m2).T)
    
    w0 = -1/2*np.dot((m1+m2),W) #这个是阈值
    return W,w0
    
def OFK(train_y,K):
    (m,) = train_y.shape
    train_Y = np.zeros((m,len(K)))
    for j in range(0,m):
        for i in range(0,len(K)):
            if train_y[j] == K[i]:
                train_Y[j,i] = 1
    return train_Y

    
# Train_x->60000*784      train_y->60000*1
# Test_x->10000*784       test_y->10000*1
filename_train_x = 'train-images-idx3-ubyte'
# train_x = Analytic_fun(filename_train_x)
Train_x = loadImageSet(filename_train_x)
filename_train_y = 'train-labels-idx1-ubyte'
#Train_y = Analytic_fun(filename_train_y)
Train_y = loadLabelSet(filename_train_y)
filename_test_x = 't10k-images-idx3-ubyte'
Test_x = loadImageSet(filename_test_x)
filename_test_y = 't10k-labels-idx1-ubyte'
Test_y = loadLabelSet(filename_test_y)


