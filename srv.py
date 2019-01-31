
# -*- coding: UTF-8 -*-

import io
import os
import re
import math
import time
import json
import logging
import datetime
from logging.handlers import RotatingFileHandler
from flask import Flask, request, session, g, redirect, url_for, \
    abort, render_template, flash, make_response, jsonify, json, \
    send_from_directory
from flask_cors import CORS
from flask.views import MethodView
from uadetector.flask import UADetector
from decimal import Decimal
from urllib.error import HTTPError
from urllib.parse import quote, unquote

application = Flask(__name__)
UADetector(application)

CORS(application, resources=r'/api/*')

import os
from PIL import Image
import numpy as np
import tensorflow as tf

data_dir = "data" # 트레이닝 이미지 저장 폴더
data_dirt = "datat" # 테스트 이미지 저장 폴더
train = False# 트레이닝 or 테스트 선택하기
model_path = "model/image_model" #model 저장경로

IMG_FILENAME = '{}/0_1.jpg'.format(data_dirt)

# 이미지 폴더로부터 이미지들을 load하고 numpy에 저장한다.
# 이미지 filename이 1_40.jpg 일 때, Lable=1 이다.
def read_data(data_dir):
    datas = []
    labels = []
    fpaths = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        fpaths.append(fpath)
        image = Image.open(fpath)


        data = np.array(image) / 255.0
        label = int(fname.split("_")[0])
        datas.append(data)
        labels.append(label)

    datas = np.array(datas)
    labels = np.array(labels)


    print("shape of datas: {}\tshape of labels: {}".format(datas.shape, labels.shape))
    return fpaths, datas, labels



@application.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(application.root_path, 'static'),
                               '/images/favicon.ico', mimetype='image/vnd.microsoft.icon')

@application.route('/api/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        if 'file' not in request.files:
            return json.dumps({
                'msg': 'No file'
            })
        file = request.files['file']
        print (file.filename)

        file.save(IMG_FILENAME)

        fpaths, datas, labels = read_data(data_dir) #트레이닝 데이터
        fpathst, datast, labelst = read_data(data_dirt) #테스트 데이터


        # 이미지 class 개수
        num_classes = len(set(labels))

        # Placeholder에 input데이터와 label를 저장한다.
        datas_placeholder = tf.placeholder(tf.float32, [None, 32, 32, 3])
        labels_placeholder = tf.placeholder(tf.int32, [None])

        # Placeholder에 DropOut 파라미터를 저장한다.트레이닝할 때 0.25，테스트 할 때 0.
        dropout_placeholdr = tf.placeholder(tf.float32)

        # 첫번째 Convolutional Layer0
        # input은 이미지=32x32x3, filters 필터개수 =20, kernal_size 필터사이즈=5x5，stride 이동=1x1
        conv0 = tf.layers.conv2d(datas_placeholder, 20, 5, activation=tf.nn.relu)
        print(conv0)
        # output size= [(W - F + 2P)/S] +1 = (32-5)/1+1 =28
        # shape=(?, 28, 28, 20)

        # 첫번째 max-pooling0 사이즈를 1/4로 줄인다
        # input=28x28x20, pooling size=2x2，stride이동=2x2
        pool0 = tf.layers.max_pooling2d(conv0, [2, 2], [2, 2])
        print(pool0)
        # output size= W = [(W - F )/S] +1 = [(28-2)/2]+1 =14
        # shape=(?, 14, 14, 20)

        # 두번째 Convolutional Layer1
        # input=14x14x20, filters 필터개수=40, kernal_size 필터사이즈=4x4，stride 이동=1x1
        conv1 = tf.layers.conv2d(pool0, 40, 4, activation=tf.nn.relu)
        print(conv1)
        # output size= [(W - F + 2P)/S] +1 = (14-4)/1+1 =11
        # shape=(?, 11, 11, 40)

        # 두번째 max-pooling1 사이즈를 1/4로 줄인다
        # input=11x11x40, pooling size=2x2，stride 이동=2x2
        pool1 = tf.layers.max_pooling2d(conv1, [2, 2], [2, 2])
        print(pool1)
        # output size= W = [(W - F )/S] +1 = [(11-2)/2]+1 =5.5
        # shape=(?, 5, 5, 40)

        # 3-D vector을 1-D vector로 변환한다.
        flatten = tf.layers.flatten(pool1)
        print(flatten)
        # output size=5x5x40=1000
        # shape=(?, 1000)

        # fully-connected layer
        fc = tf.layers.dense(flatten, 400, activation=tf.nn.relu)
        print(fc)
        # shape=(?, 400)

        # DropOut를 추가하여 overfitting을 방지한다.
        dropout_fc = tf.layers.dropout(fc, dropout_placeholdr)
        print(dropout_fc)
        # shape=(?, 400)

        # output=4개 class
        logits = tf.layers.dense(dropout_fc, num_classes)
        print(logits)
        # shape=(?, 4)

        predicted_labels = tf.arg_max(logits, 1)


        # loss를 정의한다.
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.one_hot(labels_placeholder, num_classes), #실제 label
            logits=logits #트레이닝 output label
        )

        mean_loss = tf.reduce_mean(losses)

        # optimizer를 정의한다.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(losses)


        saver = tf.train.Saver()



        with tf.Session() as sess:
            print("testing")
            # 트레이닝한 model을 불러온다
            saver.restore(sess, model_path)
            print("from {} loaded".format(model_path))
            # label의 각 의미
            label_name_dict = {
                0: "airplane",
                1: "car",
                2: "bird",
            }
            # placeholder에 데이터를 넣는다.
            test_feed_dict = {
                datas_placeholder: datast,
                labels_placeholder: labelst,
                dropout_placeholdr: 0
            }
            predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)

            for v in predicted_labels_val:
                print (v)

            v = predicted_labels_val[0]

            ###############################
            # 여기 수정해주세요
            if v == 0:
                object_name = "비행기"
            elif v == 1:
                object_name = "자동차"
            elif v == 2:
                object_name = "새"
            ################################

            return json.dumps({
                'msg': 'success',
                'object_name': object_name
            })

@application.route('/<path:path>', methods=['GET'])
def router(path):
    print (path)
    return render_template(path)

if __name__ == '__main__':
    handler = RotatingFileHandler('server.log', maxBytes=10000, backupCount=2)
    handler.setFormatter(logging.Formatter(
        '[%(levelname)s:%(name)s: %(message)s in %(asctime)s; %(filename)s:%(lineno)d'
    ))
    handler.setLevel(logging.DEBUG)
    #application.debug = False
    application.logger.addHandler(handler)
    application.run(host='0.0.0.0', port='5050')
