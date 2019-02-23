'''Train MobileNet on the CIFAR10 small iamges dataset.
Inspired by:
https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py
'''
# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import keras
from keras.datasets import cifar10
from keras.applications.mobilenetv2 import MobileNetV2

batch_size = 32
num_classes = 10
epochs = 50

# Get the data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Preprocess the images.
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Get the model and compile it.
img_input = keras.layers.Input(shape=(32, 32, 3))
model = MobileNetV2(input_tensor=img_input,
                  weights = None,
                  classes=num_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Training model.")
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          verbose=1)
