"""
Download Fashion MNIST dataset
"""

import keras


class FashionMnistDataset:
    def __init__(self):
        self.dataset = keras.datasets.fashion_mnist
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def create_dataset(self):
        # download fashion mnist
        (train_X, train_y), (test_X, test_y) = self.dataset.load_data()

        # scale images from 0 to 1
        train_X = train_X / 255.0
        test_X = test_X / 255.0

        # reshape images for model input
        train_X = train_X.reshape((60000, 28, 28, 1))
        test_X = test_X.reshape((10000, 28, 28, 1))

        return (train_X, train_y), (test_X, test_y)
