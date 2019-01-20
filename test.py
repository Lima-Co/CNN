import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf


IMG_SIZE = 50
LR = 1e-3   # change the learnning rate
MODEL_NAME = 'chatbot-{}-{}.model'.format(LR, '6conv-basic')
# just so we remember which saved model is which, sizes must match

test_data = np.load('data/test_data.npy')

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')
model.load(MODEL_NAME)
print('model loaded!')


for filename in os.listdir("test"):
    print(filename)
    path= "test"+ "\\" +filename
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    data = np.array(img)
    data = data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([data])[0]
    print(model_out)
    if np.argmax(model_out) == 0:
        print('자장면')
    else:
        print('짬뽕')
    print('\n')

#
# for num,data in enumerate(test_data):
#     # Zzazang: [1,0]
#     # Zambbong: [0,1]
#
#     img_num = data[1]
#     img_data = data[0]
#
#     orig = img_data
#     data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
#
#     model_out = model.predict([data])[0]
#
#     print (model_out)
#
#     if np.argmax(model_out) == 1: print ('Zzazang')
#     else: print ('Zambbong')
#
#
