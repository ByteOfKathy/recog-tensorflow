import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
matplotlib.use('tkagg')

train_data = './trainingData/train'
test_data = './trainingData/test'

def one_hot_label(img):
    ohl = None
    label = img.split('.')[0]
    if label == 'eyes':
        ohl = np.array([1, 0])
    elif label == 'nose':
        ohl = np.array([0, 1])
    return ohl

def train_data_with_label():
    train_imgs = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        train_imgs.append([np.array(img), one_hot_label(i)])
    shuffle(train_imgs)
    return train_imgs

def test_data_with_label():
    test_imgs = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (64, 64))
        test_imgs.append([np.array(img), one_hot_label(i)])
    return test_imgs


training_images = train_data_with_label()
testing_images = test_data_with_label()

tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[1] for i in training_images])

tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,64,64,1)
tst_lbl_data = np.array([i[1] for i in testing_images])

model = Sequential()
model.add(InputLayer(input_shape = [64, 64, 1]))

model.add(Conv2D(filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Conv2D(filters = 50, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Conv2D(filters = 80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
model.add(MaxPool2D(pool_size = 5, padding = 'same'))

model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(rate = 0.5))
model.add(Dense(2, activation = 'softmax'))

model.compile(optimizer = Adam(lr = 1e-3), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(x = tr_img_data, y = tr_lbl_data, epochs = 50, batch_size = 128)
model.save('./models/eyesNose.h5')
model.summary()