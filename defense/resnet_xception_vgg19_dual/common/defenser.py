import keras
import os

from pretrained_weight_downloader import PretrainedModelDefinition
from keras.layers import Input, Lambda


class Defenser:
    def __init__(self, model, image_input, output, input_preprocessor):
        self.__model = model
        self.__image_input = image_input
        self.__output = output
        self.__input_preprocessor = input_preprocessor
        self.__prediction_fn = keras.backend.function(
            [image_input, keras.backend.learning_phase()],
            [output],
        )

    @staticmethod
    def load_h5(model_name, path):
        model_definition = PretrainedModelDefinition(model_name)
        model = keras.models.load_model(path)
        return Defenser(
            model, model.input, model.output, model_definition.get_input_preprocessor(),
        )

    @staticmethod
    def create(model_name, base_dir=None, pretrained_trainable=True, weight_path=None, add_unet=False):
        model_definition = PretrainedModelDefinition(model_name)

        image_input = Input(shape=(299, 299, 3))
        if add_unet:
            input_tensor = Lambda(lambda image: keras.backend.tf.image.resize_images(image, (256, 256)))(image_input)

        size = model_definition.get_image_size()
        input_tensor = Lambda(lambda image: keras.backend.tf.image.resize_images(image, (size, size)))(image_input)

        model = model_definition.get_model()(
            include_top=True, weights=None, input_tensor=input_tensor,
            input_shape=None, pooling=None, classes=1000
        )

        if base_dir is not None:
            model.load_weights(os.path.join(base_dir, model_definition.get_weight()))
        if weight_path is not None:
            model.load_weights(weight_path)

        for layer in model.layers[1:]:
            layer.trainable = pretrained_trainable
        output = model.layers[-1].output

        return Defenser(model, image_input, output, model_definition.get_input_preprocessor())

    def get_model(self):
        return self.__model

    def get_image_input(self):
        return self.__image_input

    def get_output(self):
        return self.__output

    def get_input_preprocessor(self):
        return self.__input_preprocessor

    def get_prediction_fn(self):
        return self.__prediction_fn

    def predict(self, images):
        x = self.__input_preprocessor(images.copy())
        prediction = self.__prediction_fn([x, 0])[0]
        return prediction
