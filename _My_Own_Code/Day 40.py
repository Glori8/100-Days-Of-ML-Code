# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 10:03:46 2019

@author: BME207_1
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

DATADIR = "D:\\kagglecatsanddogs_3367a\\PetImages"
CATEGORIES = ["Dog","Cat"]

for category in CATEGORIES: 
    path = os.path.join(DATADIR,category)  # 创建路径
    for img in os.listdir(path):  # 迭代遍历每个图片
        img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # 转化成array
        plt.imshow(img_array, cmap='gray')  # 转换成图像展示
        plt.show()  # display!
        break  # 作为演示只展示一张，所以直接break了
    break  #同上

print(img_array)
print(img_array.shape)

IMG_SIZE = 100
new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
plt.imshow(new_array,cmap = 'gray')
plt.show()

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR,category)
        class_num = CATEGORIES.index(category)
        
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()
print(len(training_data))           

import random
random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])

X = []
y = []
for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1,IMG_SIZE,IMG_SIZE,1))
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)

import pickle
pickle_out = open("../datasets/X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("../datasets/y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("../datasets/X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("../datasets/y.pickle","rb")
y = pickle.load(pickle_in)













