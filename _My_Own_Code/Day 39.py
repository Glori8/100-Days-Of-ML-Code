# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 10:46:34 2019

@author: BME207_1
"""

import tensorflow.keras as keras
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

import matplotlib.pyplot as plt
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()
print(y_train[0])

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

model = tf.keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=(28,28)))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(128,activation=tf.nn.relu))
model.add(keras.layers.Dense(10,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train,y_train,epochs=3)

val_loss,val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

predictions = model.predict(x_test)
print(predictions)

import numpy as np
print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()

#保存模型
model.save('epic_num_reader.model')
#加载保存的模型
new_model = tf.keras.models.load_model('epic_num_reader.model')
#测试保存的模型
predictions = new_model.predict(x_test)
print(np.argmax(predictions[0]))














