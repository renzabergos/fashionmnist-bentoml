"""
Script for training the Fashion MNIST dataset
"""

import os
from data.fashion_mnist import FashionMnistDataset
from models.cnn import CNN

ROOT_DIR = os.getcwd()


def main():
    # Retrieve dataset
    fashion_mnist = FashionMnistDataset()
    (train_X, train_y), (test_X, test_y) = fashion_mnist.create_dataset()

    # Assemble model
    cnn = CNN()
    model = cnn.make_model(metrics=['accuracy'])

    # Train model
    model.fit(train_X, train_y, batch_size=64, epochs=3)

    # Evaluate model
    test_loss, test_acc = model.evaluate(test_X, test_y)
    print('Test loss:', test_loss)
    print('Test accuracy:', test_acc)

    # Save model
    model_name = "keras_CNN3_fashion_mnist"
    model.save(ROOT_DIR + '/models/' + model_name + '.h5')


if __name__ == "__main__":
    main()
