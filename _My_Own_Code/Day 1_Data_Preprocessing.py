# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 10:15:46 2019

@author: BME207_1
"""

import numpy as np
import pandas as pd

dataset = pd.read_csv("D:/100-Days-Of-ML-Code/datasets/Data.csv")
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print("Step 2: Importing dataset")
print("X")
print(X)


print("Y")
print(Y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])
print("---------------------")
print("Step 3: Handling the missing data")
print("step2")
print("X")
print(X)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)
print("---------------------")
print("Step 4: Encoding categorical data")
print("X")
print(X)
print("Y")
print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
print("---------------------")
print("Step 5: Splitting the datasets into training sets and Test sets")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#根据对之前部分trainData进行fit的整体指标，
#对剩余的数据（testData）使用同样的均值、方差、最大最小值
#等指标进行转换transform(testData)，从而保证train、test处理方式相同。
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("---------------------")
print("Step 6: Feature Scaling")
print("X_train")
print(X_train)
print("X_test")
print(X_test)

























