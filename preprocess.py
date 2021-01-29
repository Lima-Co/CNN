import cv2                                
import numpy as np                 
import os                                 
from random import shuffle       
from tqdm import tqdm           

TRAIN_DIR = './train'  # Attention! /
TEST_DIR = './test'   # Attention! /
IMG_SIZE = 50

def label_img(img):# add label to each image
    word_label = img.split('_')[0]
    # print(word_label)
    # Zzazang: [1,0]
    # Zambbong: [0,1]
    if word_label == '1':
        return [1,0]
    elif word_label == '2':
        return [0,1]
    

def create_train_data(): # create training data, and next time just load train_data.npy
    training_data = []
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR,img)
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        training_data.append([np.array(img),np.array(label)])
    shuffle(training_data)
    np.save('data/train_data.npy', training_data)
    return training_data

def create_test_data():# create testing data, similar to create train data
    testing_data = []
    # 100.jpg
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img_num = img.split('.')[0]
        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)  #color
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append([np.array(img), img_num])
    shuffle(testing_data)  
    np.save('data/test_data.npy', testing_data)
    return testing_data


if __name__=='__main__':
    train_data = create_train_data()
    test_data = create_test_data()