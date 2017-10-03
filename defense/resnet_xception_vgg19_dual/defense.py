import numpy as np

from common.defenser import Defenser
from common.nips_util import NipsUtil


def run():
    import keras
    nips_util = NipsUtil()
    predictions = {}

    def predict(defenser):
        prediction = nips_util.simple_defenser_predict_labels_score([defenser])
        for key, value in prediction.items():
            if key not in predictions:
                predictions[key] = np.zeros(value.shape)
            predictions[key] += value
        keras.backend.clear_session()

    predict(Defenser.create('vgg19', 'common').predict)
    predict(Defenser.create('resnet', 'common').predict)
    predict(Defenser.create('xception', 'common').predict)
    predict(Defenser.create('vgg19', weight_path='vgg19.h5').predict)
    predict(Defenser.create('resnet', weight_path='resnet.h5').predict)
    predict(Defenser.create('xception', weight_path='xception.h5').predict)

    labels = {key: np.argmax(value) for key, value in predictions.items()}
    nips_util.write_defense(labels)

if __name__ == '__main__':
    run()
