"""
MIT License

Original Copyright (c) 2017 Google Inc., OpenAI and Pennsylvania State University
Modification Copyright (c) 2017 Chihiro Komaki

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""
"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from cleverhans.attacks import FastGradientMethod
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.slim.nets import inception

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path', '', 'Path to checkpoint for inception network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_string(
    'step', '', 'Output directory with images.')

tf.flags.DEFINE_string(
    'step_n', '', 'Output directory with images.')

tf.flags.DEFINE_string(
    'npy_dir', '', 'Output directory with images.')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  values = np.zeros(batch_shape)
  idx = 0
  batch_size = batch_shape[0]
  found = 0
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath) as f:
      image = np.array(Image.open(f).convert('RGB')).astype(np.float) / 255.0
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    images[idx, :, :, :] = image * 2.0 - 1.0
    filenames.append(os.path.basename(filepath))
    npy_path = os.path.join(FLAGS.npy_dir, os.path.basename(filepath) + '.npy')
    if os.path.exists(npy_path):
        value = np.load(npy_path)
        found += 1
    else:
        value = images[idx].copy()
    values[idx, :, :, :] = value
    idx += 1
    if idx == batch_size:
      yield filenames, images, values
      filenames = []
      values = np.zeros(batch_shape)
      images = np.zeros(batch_shape)
      idx = 0
  print("found: %d" % found)
  if idx > 0:
    yield filenames, images, values


def save_images(values, filenames, output_dir):
  """Saves images to the output directory.

  Args:
    images: array with minibatch of images
    filenames: list of filenames without path
      If number of file names in this list less than number of images in
      the minibatch then only first len(filenames) images will be saved.
    output_dir: directory where to save images
  """
  for i, filename in enumerate(filenames):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # so rescale them back to [0, 1].
    with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
      img = (((values[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
      Image.fromarray(img).save(f, format='PNG')
    np.save(os.path.join(FLAGS.npy_dir, filename + '.npy'), values[i])


class InceptionModel(object):
  """Model class for CleverHans library."""

  def __init__(self, num_classes):
    self.num_classes = num_classes
    self.built = False

  def __call__(self, x_input):
    """Constructs model and return probabilities for given input."""
    reuse = True if self.built else None


    if not False:
      with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
            x_input, num_classes=self.num_classes, is_training=False,
            reuse=reuse)
    else:
      import inception_resnet_v2
      with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        _, end_points = inception_resnet_v2.inception_resnet_v2(
          x_input, num_classes=self.num_classes, is_training=False,
          reuse=reuse)

    self.built = True
    output = end_points['Predictions']
    # Strip off the extra reshape op at the output
    probs = output.op.inputs[0]
    return probs


def main(_):
  # Images for inception classifier are normalized to be in [-1, 1] interval,
  # eps is a difference between pixels so it should be in [0, 2] interval.
  # Renormalizing epsilon from [0, 255] to [0, 2].
  eps = 2.0 * FLAGS.max_epsilon / 255.0
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    model = InceptionModel(num_classes)

    fgsm = FastGradientMethod(model)
    x_adv = fgsm.generate(x_input, eps=eps)

    # Run computation
    saver = tf.train.Saver(slim.get_model_variables())
    print("check point: %s" % FLAGS.checkpoint_path)
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=tf.train.Scaffold(saver=saver),
        checkpoint_filename_with_path=FLAGS.checkpoint_path,
        master=FLAGS.master)

    with tf.train.MonitoredSession(session_creator=session_creator) as sess:
      for filenames, images, values in load_images(FLAGS.input_dir, batch_shape):
        max_epsilon = FLAGS.max_epsilon * 2 / 255.0
        max_clip = np.clip(images + eps, -1, 1.0)
        min_clip = np.clip(images - eps, -1, 1.0)
        adv_values = sess.run(x_adv, feed_dict={x_input: values})
        grad = adv_values - values
        grad = grad / (np.std(grad, axis=(1, 2, 3), keepdims=True) + 1e-15)
        values = values + float(float(FLAGS.step_n) - float(FLAGS.step)) / float(FLAGS.step_n) * 0.5 * grad * max_epsilon
        values = np.maximum(values, min_clip)
        values = np.minimum(values, max_clip)
        save_images(values, filenames, FLAGS.output_dir)


if __name__ == '__main__':
  tf.app.run()
