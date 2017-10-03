from keras import optimizers, losses


class Trainer:
    def __init__(self, defenser):
        print("Creating trainer")
        self.__model = defenser.get_model()
        self.__model.compile(
            optimizer=optimizers.SGD(lr=1e-4, decay=0, momentum=0.9, nesterov=True),
            loss=losses.categorical_crossentropy,
            metrics=None,
            loss_weights=None, sample_weight_mode=None
        )
        self.__defenser = defenser
        print("Created trainer")

    def save_weights(self, path):
        self.__model.save_weights(path)

    def predict(self, x):
        return self.__defenser.predict(x)

    def train(self, x, y):
        x = self.__defenser.get_input_preprocessor()(x.copy())
        self.__model.train_on_batch(x=x, y=y)
