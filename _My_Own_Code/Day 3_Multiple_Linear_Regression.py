# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:52:12 2019

@author: BME207_1
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("D:/100-Days-Of-ML-Code/datasets/50_Startups.csv")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:,3] = labelencoder.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

y_pred = regressor.predict(X_test)
from sklearn.metrics import r2_score
print(r2_score(Y_test, y_pred))