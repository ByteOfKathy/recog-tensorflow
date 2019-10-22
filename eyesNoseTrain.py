import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline

train_data = './trainingData/train'
test_data = './trainingData/test'

def one_hot_label(img):
    label = img.split('.')[0]
    if label == 'eye':
        ohl = np.array([1, 0])
    elif label == 'nose':
        ohl = np.array([0, 1])
    return ohl

def train_data_with_label():
    train_imgs = []
    for i in tqdm(os.listdir(train_data))