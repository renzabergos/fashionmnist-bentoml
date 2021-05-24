from typing import List

import numpy as np
from PIL import Image
from bentoml import api, artifacts, env, BentoService
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.adapters import ImageInput

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@env(pip_packages=['keras==2.3.1', 'tensorflow==1.14.0', 'pillow', 'numpy', 'h5py==2.10.0'])
@artifacts([KerasModelArtifact('classifier')])
class KerasFashionMnistService(BentoService):
    @api(input=ImageInput(pilmode='L'), batch=True)
    def predict(self, imgs: List[np.ndarray]) -> List[str]:
        inputs = []
        for img in imgs:
            # resize to 28x28
            img = Image.fromarray(img).resize((28, 28))
            # reshape to correct model input
            img = np.array(img.getdata()).reshape((28, 28, 1))
            inputs.append(img)
        inputs = np.stack(inputs)
        class_idxs = self.artifacts.classifier.predict_classes(inputs)
        return [class_names[class_idx] for class_idx in class_idxs]
