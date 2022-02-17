# -*- coding: utf-8 -*-
"""
演示内容：量纲的特征缩放
（两种方法：标准化缩放法和区间缩放法。每种方法举了两个例子：简单二维矩阵和iris数据集）
"""
#方法1：标准化缩放法 例1：对简单示例二维矩阵的列数据进行
from sklearn import preprocessing   
import numpy as np  
#采用numpy的array表示，因为要用到其mean等函数，而list没有这些函数
X = np.array([[0, 0], 
        [0, 0], 
        [100, 1], 
        [1, 1]])  
# calculate mean  
X_mean = X.mean(axis=0)  
# calculate variance   
X_std = X.std(axis=0)  
#print (X_std)
# standardize X  
X1 = (X-X_mean)/X_std
print (X1)
print ("")
 
# we can also use function preprocessing.scale to standardize X  
X_scale = preprocessing.scale(X)  
print (X_scale)
 
 
#方法1： 标准化缩放法 例2：对iris数据二维矩阵的列数据进行。这次采用一个集成的方法StandardScaler
from sklearn import datasets
iris = datasets.load_iris()
X_scale = preprocessing.scale(iris.data)  
print (X_scale)
 
#方法2： 区间缩放法 例3：对简单示例二维矩阵的列数据进行
from sklearn.preprocessing import MinMaxScaler
 
data = [[0, 0], 
        [0, 0], 
        [100, 1], 
        [1, 1]]
 
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.transform(data))
 
#方法2： 区间缩放法 例4：对iris数据二维矩阵的列数据进行
from sklearn.preprocessing import MinMaxScaler
 
data = iris.data
 
scaler = MinMaxScaler()
print(scaler.fit(data))
print(scaler.transform(data))