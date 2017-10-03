import keras
import numpy as np

from defenser import Defenser
from keras.layers import Input


class Attacker:
    def __init__(self, defenser, predictions=None):
        """
        @type defenser: Defenser
        """
        label_input = Input(shape=(1000,))
        loss = keras.losses.categorical_crossentropy(label_input, defenser.get_output())
        self.__defenser = defenser
        self.__adversarial_fn = keras.backend.function(
            [defenser.get_image_input(), label_input, keras.backend.learning_phase()],
            keras.backend.gradients(loss, [defenser.get_image_input()])
        )
        self.__input_preprocessor = defenser.get_input_preprocessor()
        self.__predictions = predictions

    def get_gradients_init_most(self, images, image_names):
        target_label_indexes = map(lambda image_name_: self.__predictions[image_name_], image_names)
        return self.get_gradients(images, target_label_indexes)

    def get_gradients_most(self, images, image_names):
        prediction = self.__defenser.predict(images.copy())
        target_label_indexes = np.argmax(prediction, axis=1)
        return self.get_gradients(images, target_label_indexes)

    def get_gradients_least(self, images, image_names):
        prediction = self.__defenser.predict(images.copy())
        target_label_indexes = np.argmin(prediction, axis=1)
        return self.get_gradients(images, target_label_indexes)

    def get_gradients(self, images, target_label_indexes):
        labels = np.zeros((len(images), 1000))
        for i in range(len(target_label_indexes)):
            labels[i][(target_label_indexes[i]) % len(labels[i])] = 1

        x = self.__defenser.get_input_preprocessor()(images.copy())
        gradients = self.__adversarial_fn([x, labels, 0])[0]
        return gradients / (np.std(gradients, axis=(1, 2, 3), keepdims=True) + 1e-15)
