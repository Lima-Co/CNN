
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

IMG_FILENAME = 'TMP_IMAGE.jpg'
IMG_SIZE = 50
LR = 1e-3   # change the learnning rate
MODEL_NAME = 'chatbot-{}-{}.model'.format(LR, '6conv-basic')
# just so we remember which saved model is which, sizes must match

test_data = np.load('data/test_data.npy', allow_pickle=True)

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

application = Flask(__name__)
UADetector(application)

CORS(application, resources=r'/api/*')

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

        img = cv2.imread(IMG_FILENAME, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data = np.array(img)
        data = data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]
        object_name = '짬뽕'
        if np.argmax(model_out) == 0:
            object_name = '자장면'

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
