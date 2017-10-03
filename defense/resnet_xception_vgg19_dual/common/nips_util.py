import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser
from numpy import random
from scipy.misc import imsave


class NipsUtil:
    def write(self, predicted_labels):
        """
        @type predicted_labels: dict[str, int]
        """
        with open(self.__args.output_file, 'w') as fp:
            for image_name, label in predicted_labels.items():
                fp.write('%s,%d\n' % (image_name, label + 1))

    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument('--input-dir', dest='input_dir', type=str)
        parser.add_argument('--output-file', dest='output_file', type=str)
        parser.add_argument('--output-dir', dest='output_dir', type=str)
        parser.add_argument('--max-epsilon', dest='max_epsilon', type=int)
        parser.add_argument('--fall11', dest='fall11_path', type=str)
        self.__args = parser.parse_args()

    def get_image_names(self):
        names = filter(lambda path: path.endswith('.png'), os.listdir(self.__args.input_dir))
        # names = filter(lambda _: random.randint(20) == 0, names)
        return names

    def read_target_classes(self):
        path = os.path.join(self.__args.input_dir, 'target_class.csv')
        df = pd.read_csv(path, header=None, names=['name', 'target'])
        df['target'] = df['target'].map(lambda target_: int(target_) - 1)
        df['target'] = df['target'].map(lambda target_: target_ if 0 <= target_ < 1000 else 0)
        return df.set_index('name')['target'].to_dict()

    def get_defense_output_path(self):
        return self.__args.output_file

    def get_attack_output_path(self, image_name):
        return os.path.join(self.__args.output_dir, image_name)

    def get_max_epsilon(self):
        return self.__args.max_epsilon

    def read_image(self, path):
        from keras.preprocessing import image
        return image.img_to_array(image.load_img(path))

    def read_images(self, image_names):
        return np.array([
            self.read_image(os.path.join(self.__args.input_dir, image_name))
            for image_name
            in image_names
        ]).astype(np.float64)

    def clip(self, images, base_images):
        images = np.clip(images, base_images - self.__args.max_epsilon, base_images + self.__args.max_epsilon)
        images = np.clip(images, 0, 255)
        return images

    def write_attack(self, images, image_names):
        for image, image_name in zip(images, image_names):
            imsave(os.path.join(self.__args.output_dir, image_name), image)

    def write_defense(self, predicted_labels):
        """
        @type predicted_labels: dict[str, int]
        """
        with open(self.__args.output_file, 'w') as fp:
            for image_name, label in predicted_labels.items():
                fp.write('%s,%d\n' % (image_name, label + 1))

    def simple_attacker(self, epoch, compute_gradient_list, apply_gradient):
        all_image_names = self.get_image_names()

        batch_size = 10
        for begin in range(0, len(all_image_names), batch_size):
            image_names = all_image_names[begin: begin + batch_size]
            base_images = self.read_images(image_names)
            images = base_images.copy()
            for i in range(epoch):
                gradients = compute_gradient_list[i % len(compute_gradient_list)](images, image_names)
                images = apply_gradient(images, gradients, i)
                images = self.clip(images, base_images)
            self.write_attack(images, image_names)

    def simple_target_attacker(self, epoch, compute_gradient_list, apply_gradient):
        all_target_classes = self.read_target_classes()
        all_image_names = self.get_image_names()
        self.println("image names: %d" % len(all_image_names))

        batch_size = 10
        for begin in range(0, len(all_image_names), batch_size):
            if batch_size % 100 == 0:
                self.println("begin: %d" % begin)
            image_names = all_image_names[begin: begin + batch_size]
            target_classes = map(lambda image_name_: all_target_classes[image_name_], image_names)
            base_images = self.read_images(image_names)
            images = base_images.copy()
            for i in range(epoch):
                gradients = compute_gradient_list[i % len(compute_gradient_list)](images, target_classes)
                images = apply_gradient(images, gradients, i)
                images = self.clip(images, base_images)
            self.write_attack(images, image_names)

    def simple_defenser_predict_labels(self, predicts):
        if not isinstance(predicts, list):
            predicts = [predicts]

        all_image_names = self.get_image_names()
        self.println("image names: %d" % len(all_image_names))

        predicted_labels = {}
        batch_size = 10
        for begin in range(0, len(all_image_names), batch_size):
            if batch_size % 100 == 0:
                self.println("begin: %d" % begin)
            image_names = all_image_names[begin: begin + batch_size]
            images = self.read_images(image_names)
            predictions = np.sum([predict(images) for predict in predicts], axis=0)
            for image_name, prediction in zip(image_names, predictions):
                predicted_labels[image_name] = np.argmax(prediction)
        return predicted_labels

    def simple_defenser_predict_labels_score(self, predicts):
        if not isinstance(predicts, list):
            predicts = [predicts]

        all_image_names = self.get_image_names()
        self.println("image names: %d" % len(all_image_names))

        predicted_labels = {}
        batch_size = 10
        for begin in range(0, len(all_image_names), batch_size):
            if batch_size % 100 == 0:
                self.println("begin: %d" % begin)
            image_names = all_image_names[begin: begin + batch_size]
            images = self.read_images(image_names)
            predictions = np.sum([predict(images) for predict in predicts], axis=0)
            for image_name, prediction in zip(image_names, predictions):
                predicted_labels[image_name] = prediction
        return predicted_labels

    def println(self, message):
        import sys
        print(message)
        sys.stdout.flush()

    def simple_trainer_train(self, trainer):
        image_to_label = pd.read_csv(self.__args.fall11_path).set_index('name')['imagenet_index'].to_dict()
        # print image_to_label
        paths = {}
        for attack_type in ['attacks_output', 'targeted_attacks_output']:
            self.println(os.path.join(self.__args.input_dir, attack_type))
            for attacker_name in os.listdir(os.path.join(self.__args.input_dir, attack_type)):
                print attacker_name
                found = 0
                attacker_path = os.path.join(self.__args.input_dir, attack_type, attacker_name)
                if not os.path.isdir(attacker_path):
                    continue
                for image_name in os.listdir(attacker_path):
                    label = image_to_label.get(image_name[:-4], None)
                    if label is None:
                        continue
                    image_path = os.path.join(attacker_path, image_name)
                    if label in [134, 517]:
                        paths.setdefault(134, []).append(image_path)
                        paths.setdefault(517, []).append(image_path)
                    else:
                        paths.setdefault(label, []).append(image_path)
                    found += 1
                self.println("attack %s: %d" % (attacker_name, found))
        self.println("label num: %d" % len(paths.keys()))
        for i in range(1000):
            if i not in paths:
                self.println(i)

        def get_random(l_):
            return l_[random.randint(len(l_))]

        batch_size = 12
        for step in range(40):
            score = 0
            for _ in range(100):
                labels = [get_random(list(paths.keys())) for _ in range(batch_size)]
                label_onehots = np.zeros((batch_size, 1000))
                for label_i, label in enumerate(labels):
                    label_onehots[label_i, label] = 1

                # print get_random(paths[labels[0]])
                images = np.array([self.read_image(get_random(paths[label])) for label in labels])
                predictions = trainer.predict(images)
                trainer.train(images, label_onehots)
                score += np.mean((predictions - label_onehots) ** 2)
            self.println("%4d: %f" % (step, score))

        self.println("saving to: %s" % self.__args.output_file)
        trainer.save_weights(self.__args.output_file)
