"""
Create BentoML Service model artifacts
"""

import os
from deployments.keras_fashion_mnist import KerasFashionMnistService
from data.fashion_mnist import FashionMnistDataset
from keras.models import load_model
import bentoml

ROOT_DIR = os.getcwd()


def main():
    # Load model
    model_name = "keras_CNN3_fashion_mnist"
    model_dir = ROOT_DIR + '/models/' + model_name + '.h5'
    print(model_dir)
    model = load_model(model_dir)
    

    bento_svc = KerasFashionMnistService()
    bento_svc.pack('classifier', model)

    saved_path = bento_svc.save()

    print(saved_path)

    # test service
    fm = FashionMnistDataset()
    (train_X, train_y), (test_X, test_y) = fm.create_dataset()
    svc = bentoml.load(saved_path)
    print(svc.predict([test_X[0].squeeze(-1)]))



if __name__ == "__main__":
    main()
