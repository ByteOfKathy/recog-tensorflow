import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

# pull data from sample set
data = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = data.load_data()
class_names = ['tee/top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot' ]

# normalize to a number between 0 & 1
train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

# try and load model
try:
    model = keras.models.load_model('./models/fashionexample.h5')
except FileNotFoundError:
    print('fashionexample.h5 not found/trained.')
    exit()

test_loss, test_acc = model.evaluate(test_imgs, test_labels)
print('loss:' + str(test_loss))

prediction = model.predict(test_imgs)

for i in range(5):
    plt.grid(False)
    plt.imshow(test_imgs[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title('prediction: ' + class_names[np.argmax(prediction[i])])
    plt.show()