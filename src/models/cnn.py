"""
Convolutional Neural Network Architecture in Keras
"""

import keras
import tensorflow as tf


METRICS = [
    keras.metrics.Accuracy(name='accuracy')
]


class CNN:
    def __init__(self):
        print("Tensorflow Version: %s" % tf.__version__)
        print("Keras Version: %s" % keras.__version__)

    def make_model(self, metrics=METRICS):
        model = keras.Sequential()

        # Must define the input shape in the first layer of the neural network
        model.add(keras.layers.Conv2D(filters=64, kernel_size=2,
                  padding='same', activation='relu', input_shape=(28, 28, 1)))
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Conv2D(filters=32, kernel_size=2,
                  padding='same', activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=2))
        model.add(keras.layers.Dropout(0.3))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(256, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=metrics)

        # print model summary
        model.summary()

        return model
