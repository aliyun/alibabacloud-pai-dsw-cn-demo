# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import oss2
import random

import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order

from mlperf_compliance import mlperf_log
import imagenet_preprocessing
import nets.resnet_model as resnet_model
import nets.resnet_model_gap as resnet_model_gap
import nets.resnet_model_teach as resnet_model_teach
import resnet_run_loop
import tensorflow_io.oss
import tensorflow_io.oss.python.ops.ossfs_ops


_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_CLASSES = 1001

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_NUM_TRAIN_FILES = 8
_SHUFFLE_BUFFER = 1500

_BASE_LR = 0.128

_ACCESS_ID = "<TBD>"
_ACCESS_KEY = "<TBD>"
_HOST = "<TBD>"
_BUCKET = "<TBD>"
_BUCKET_DIR = "<TBD>"

oss_bucket_root="oss://{}\x01id={}\x02key={}\x02host={}/".format(_BUCKET, _ACCESS_ID, _ACCESS_KEY, _HOST)

###############################################################################
# Data processing
###############################################################################
def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-00008' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00001' % i)
        for i in range(1)]


def get_filenames_oss(is_training):
  """Return filenames for dataset."""

  if is_training:
    return [
        oss_bucket_root+_BUCKET_DIR+'/train-%05d-of-00008' % i
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        oss_bucket_root+_BUCKET_DIR+'/validation-%05d-of-00001' % i
        for i in range(1)]


def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.

  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):

    image/height: 462
    image/width: 581
    image/colorspace: 'RGB'
    image/channels: 3
    image/class/label: 615
    image/class/synset: 'n03623198'
    image/class/text: 'knee pad'
    image/object/bbox/xmin: 0.1
    image/object/bbox/xmax: 0.9
    image/object/bbox/ymin: 0.2
    image/object/bbox/ymax: 0.6
    image/object/bbox/label: 615
    image/format: 'JPEG'
    image/filename: 'ILSVRC2012_val_00041207.JPEG'
    image/encoded: <JPEG encoded string>

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.

  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
      'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
                                          default_value=''),
      'image/class/label': tf.FixedLenFeature([], dtype=tf.int64,
                                              default_value=-1),
      'image/class/text': tf.FixedLenFeature([], dtype=tf.string,
                                             default_value=''),
  }
  sparse_float32 = tf.VarLenFeature(dtype=tf.float32)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_float32 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)

  return features['image/encoded'], label


def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.

  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).

  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.

  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  image_buffer, label = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image(
      image_buffer=image_buffer,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
  image = tf.cast(image, dtype)

  return image, label


def input_fn(is_training, data_dir, batch_size, num_epochs=1, num_gpus=None,
             dtype=tf.float32, mix_up=False, oss_load=False):
  """Input function which provides batches for train or eval.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    num_gpus: The number of gpus used for training.
    dtype: Data type to use for images/features

  Returns:
    A dataset that can be used for iteration.
  """
  mlperf_log.resnet_print(key=mlperf_log.INPUT_ORDER)
  if not oss_load:
    filenames = get_filenames(is_training, data_dir)
  else:
    filenames = get_filenames_oss(is_training)
  #print(filenames)
  dataset = tf.data.Dataset.from_tensor_slices(filenames)

  if is_training:
    # Shuffle the input files
    dataset = dataset.shuffle(buffer_size=_NUM_TRAIN_FILES)

  # Convert to individual records
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  return resnet_run_loop.process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=_SHUFFLE_BUFFER,
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      num_gpus=num_gpus,
      examples_per_epoch=_NUM_IMAGES['train'] if is_training else None,
      dtype=dtype,
      mix_up=mix_up
  )


def get_synth_input_fn():
  return resnet_run_loop.get_synth_input_fn(
      _DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS, _NUM_CLASSES)


###############################################################################
# Running the model
###############################################################################
class ImagenetModel(resnet_model.Model):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, final_size=2048, data_format=None, num_classes=_NUM_CLASSES,
               version=resnet_model.DEFAULT_VERSION,
               dtype=resnet_model.DEFAULT_DTYPE,
               enable_at=False):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 20:
      bottleneck = False
      final_size = final_size
    else:
      bottleneck = True
      final_size = final_size

    super(ImagenetModel, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format,
        dtype=dtype,
        enable_at=enable_at
    )


class ImagenetModelGap(resnet_model_gap.ModelGap):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, final_size=2048, data_format=None, num_classes=_NUM_CLASSES,
               pickle_model='./gap_save/res26_s4.pkl', random_init=False,
               version=resnet_model_gap.DEFAULT_VERSION,
               dtype=resnet_model_gap.DEFAULT_DTYPE,
               enable_at=False):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 20:
      bottleneck = False
      final_size = final_size #512
    else:
      bottleneck = True
      final_size = final_size #2048

    super(ImagenetModelGap, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        pickle_model=pickle_model,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format,
        dtype=dtype,
        random_init=random_init,
        enable_at=enable_at
    )


class ImagenetModelTeach(resnet_model_teach.ModelTeach):
  """Model class with appropriate defaults for Imagenet data."""

  def __init__(self, resnet_size, final_size=2048, data_format=None, num_classes=_NUM_CLASSES,
               pickle_model='./gap_save/unpruned_teacher.pkl', temp_dst=4., w_dst=4.,
               version=resnet_model_teach.DEFAULT_VERSION,
               dtype=resnet_model_teach.DEFAULT_DTYPE,
               enable_at=False, w_at=2.):
    """These are the parameters that work for Imagenet data.

    Args:
      resnet_size: The number of convolutional layers needed in the model.
      data_format: Either 'channels_first' or 'channels_last', specifying which
        data format to use when setting up the model.
      num_classes: The number of output classes needed from the model. This
        enables users to extend the same model to their own datasets.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
      dtype: The TensorFlow dtype to use for calculations.
    """

    # For bigger models, we want to use "bottleneck" layers
    if resnet_size < 20:
      bottleneck = False
      final_size = final_size #512
    else:
      bottleneck = True
      final_size = final_size #2048

    if pickle_model.startswith("oss://"):
      pickle_file = os.path.split(pickle_model)
      pickle_model = oss_bucket_root+_BUCKET_DIR+'/'+pickle_file[-1]

    super(ImagenetModelTeach, self).__init__(
        resnet_size=resnet_size,
        bottleneck=bottleneck,
        num_classes=num_classes,
        pickle_model=pickle_model,
        num_filters=64,
        kernel_size=7,
        conv_stride=2,
        first_pool_size=3,
        first_pool_stride=2,
        second_pool_size=7,
        second_pool_stride=1,
        block_sizes=_get_block_sizes(resnet_size),
        block_strides=[1, 2, 2, 2],
        final_size=final_size,
        version=version,
        data_format=data_format,
        dtype=dtype,
        temp_dst=temp_dst,
        w_dst=w_dst,
        enable_at=enable_at,
        w_at=w_at
    )


def _get_block_sizes(resnet_size):
  """Retrieve the size of each block_layer in the ResNet model.

  The number of block layers used for the Resnet model varies according
  to the size of the model. This helper grabs the layer set we want, throwing
  an error if a non-standard size has been selected.

  Args:
    resnet_size: The number of convolutional layers needed in the model.

  Returns:
    A list of block sizes to use in building the model.

  Raises:
    KeyError: if invalid resnet_size is received.
  """
  # only consider the bottleneck
  choices = {
      26: [2, 2, 2, 2],
      38: [3, 3, 3, 3],
      50: [3, 4, 6, 3],
      101: [3, 4, 23, 3],
      152: [3, 8, 36, 3],
      200: [3, 24, 36, 3]
  }

  try:
    return choices[resnet_size]
  except KeyError:
    err = ('Could not find layers for selected Resnet size.\n'
           'Size received: {}; sizes allowed: {}.'.format(
               resnet_size, choices.keys()))
    raise ValueError(err)


def imagenet_model_fn(features, labels, mode, params):
  """Our model_fn for ResNet to be used with our Estimator."""

  # Warmup and higher lr may not be valid for fine tuning with small batches
  # and smaller numbers of training images.
  base_lr = params['learn_rate'] # 64-0.128, 100-0.15, 128-0.18

  boundary_epochs=[30, 60, 80, 90]
  decay_rates=[1, 0.1, 0.01, 0.001, 1e-4]
  if params['enable_quantize'] and params['online_quantize']:
    boundary_epochs=[10, 20]
    decay_rates=[1, 0.1, 1e-2]

  if params['pickle_model'].startswith("oss://"):
    pickle_file = os.path.split(params['pickle_model'])
    params['pickle_model'] = oss_bucket_root+_BUCKET_DIR+'/'+pickle_file[-1]

  # [1, 0.1, 0.01, 0.001, 1e-4], [30, 60, 80, 90]
  learning_rate_fn = resnet_run_loop.learning_rate_with_decay(
      batch_size=params['batch_size'], batch_denom=params['batch_size'],
      num_images=_NUM_IMAGES['train'], boundary_epochs=boundary_epochs,
      decay_rates=decay_rates, base_lr=base_lr, train_epochs=params['train_epochs'],
      enable_lars=params['enable_lars'],
      enable_cos=params['enable_cos'],
      cos_alpha=params['cos_alpha'],
      warm_up=params['warm_up'])

  return resnet_run_loop.resnet_model_fn(
      features=features,
      labels=labels,
      mode=mode,
      model_class=ImagenetModelGap if params['gap_ft'] else ImagenetModel,
      model_teach=ImagenetModelTeach,
      resnet_size=params['resnet_size'],
      random_init=params['random_init'],
      final_size=params['final_size'],
      pickle_model=params['pickle_model'],
      weight_decay=params['weight_decay'],
      learning_rate_fn=learning_rate_fn,
      momentum=0.9,
      data_format=params['data_format'],
      version=params['version'],
      version_t=params['version_t'],
      loss_scale=params['loss_scale'],
      gap_train=params['gap_train'],
      gap_lambda=params['gap_lambda'],
      gap_ft=params['gap_ft'],
      loss_filter_fn=None,
      dtype=params['dtype'],
      label_smoothing=params['label_smoothing'],
      enable_lars=params['enable_lars'],
      enable_kd=params['enable_kd'],
      kd_size=params['kd_size'],
      temp_dst=params['temp_dst'],
      w_dst=params['w_dst'],
      mix_up=params['mix_up'],
      mx_mode=params['mx_mode'],
      enable_at=params['enable_at'],
      w_at=params['w_at']
  )


def main(argv):
  parser = resnet_run_loop.ResnetArgParser(
      resnet_size_choices=[18, 26, 34, 50, 101, 152, 200])

  parser.set_defaults(
       train_epochs=90,
       version=1
  )

  flags = parser.parse_args(args=argv[2:])

  if flags.oss_load:
    auth = oss2.Auth(_ACCESS_ID, _ACCESS_KEY)
    bucket = oss2.Bucket(auth, _HOST, _BUCKET)

  seed = int(argv[1])
  print('Setting random seed = ', seed)
  print('special seeding')
  mlperf_log.resnet_print(key=mlperf_log.RUN_SET_RANDOM_SEED, value=seed)
  random.seed(seed)
  tf.set_random_seed(seed)
  np.random.seed(seed)

  mlperf_log.resnet_print(key=mlperf_log.PREPROC_NUM_TRAIN_EXAMPLES,
                          value=_NUM_IMAGES['train'])
  mlperf_log.resnet_print(key=mlperf_log.PREPROC_NUM_EVAL_EXAMPLES,
                          value=_NUM_IMAGES['validation'])
  input_function = input_fn

  resnet_run_loop.resnet_main(seed,
      flags, imagenet_model_fn, input_function,
      shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS])


if __name__ == '__main__':

  tf.logging.set_verbosity(tf.logging.INFO)
  mlperf_log.ROOT_DIR_RESNET = os.path.split(os.path.abspath(__file__))[0]
  main(argv=sys.argv)
