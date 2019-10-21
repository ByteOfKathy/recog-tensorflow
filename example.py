import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# pull data from sample set
data = keras.datasets.fashion_mnist

(train_imgs, train_labels), (test_imgs, test_labels) = data.load_data()

class_names = ['tee/top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot' 
              ]