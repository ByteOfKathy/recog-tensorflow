import tensorflow as tf
from tensorflow import keras

# pull data from sample set
data = keras.datasets.fashion_mnist
(train_imgs, train_labels), (test_imgs, test_labels) = data.load_data()
class_names = ['tee/top', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot' ]

# normalize to a number between 0 & 1
train_imgs = train_imgs/255.0
test_imgs = test_imgs/255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"]
    )

# train model and save
model.fit(train_imgs, train_labels, epochs=40)
model.save('./models/fashionexample.h5')

print('actual test:')
test_loss, test_acc = model.evaluate(test_imgs, test_labels)
print('loss:' + str(test_loss) + '\naccuracy: ' + str(test_acc))
print('\ntraining complete.')