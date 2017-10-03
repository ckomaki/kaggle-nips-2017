import keras
import os
import shutil


class PretrainedModelDefinition:
    __models = {
        'inception': keras.applications.inception_v3.InceptionV3,
        'resnet': keras.applications.resnet50.ResNet50,
        'vgg16': keras.applications.vgg16.VGG16,
        'vgg19': keras.applications.vgg19.VGG19,
        'xception': keras.applications.xception.Xception,
    }

    __weights = {
        'inception': 'inception_v3_weights_tf_dim_ordering_tf_kernels.h5',
        'resnet': 'resnet50_weights_tf_dim_ordering_tf_kernels.h5',
        'vgg16': 'vgg16_weights_tf_dim_ordering_tf_kernels.h5',
        'vgg19': 'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
        'xception': 'xception_weights_tf_dim_ordering_tf_kernels.h5',
    }

    __input_preprocessors = {
        'inception': keras.applications.inception_v3.preprocess_input,
        'resnet': keras.applications.resnet50.preprocess_input,
        'vgg16': keras.applications.vgg16.preprocess_input,
        'vgg19': keras.applications.vgg19.preprocess_input,
        'xception': keras.applications.xception.preprocess_input,
    }

    __image_sizes = {
        'inception': 299,
        'resnet': 224,
        'vgg16': 224,
        'vgg19': 224,
        'xception': 299,
    }

    def __init__(self, model_name):
        self.__model_name = model_name

    def get_model(self):
        return PretrainedModelDefinition.__models[self.__model_name]

    def get_weight(self):
        return PretrainedModelDefinition.__weights[self.__model_name]

    def get_input_preprocessor(self):
        return PretrainedModelDefinition.__input_preprocessors[self.__model_name]

    def get_image_size(self):
        return self.__image_sizes[self.__model_name]

    def download_model(self, home_dir, current_dir):
        source_path = os.path.join(home_dir, '.keras', 'models', self.__weights[self.__model_name])
        if not os.path.exists(source_path):
            PretrainedModelDefinition.__models[self.__model_name](
                include_top=True, weights='imagenet', input_tensor=None,
                input_shape=None, pooling=None, classes=1000
            )
        shutil.copy(source_path, os.path.join(current_dir, self.__weights[self.__model_name]))
