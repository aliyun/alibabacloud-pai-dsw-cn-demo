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
"""Contains definitions for Residual Networks.

Residual networks ('v1' ResNets) were originally proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

The full preactivation 'v2' ResNet variant was introduced by:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv: 1603.05027

The key difference of the full preactivation 'v2' variant compared to the
'v1' variant in [1] is the use of batch normalization before every weight layer
rather than after.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import tensorflow as tf
import tensorflow_io.oss

from mlperf_compliance import mlperf_log
from mlperf_compliance import resnet_log_helper
from gap_tools import gap_finetune


_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-5
DEFAULT_VERSION = 2
DEFAULT_DTYPE = tf.float32
CASTABLE_TYPES = (tf.float16,)
ALLOWED_TYPES = (DEFAULT_DTYPE,) + CASTABLE_TYPES


################################################################################
# Convenience functions for building the ResNet model.
################################################################################
def batch_norm(inputs, training, data_format):
  """Performs a batch normalization using a standard set of parameters."""
  # We set fused=True for a significant performance boost. See
  # https://www.tensorflow.org/performance/performance_guide#common_fused_ops
  outputs = tf.layers.batch_normalization(
      inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
      scale=True, training=training, trainable=training, fused=True)

  resnet_log_helper.log_batch_norm(
      input_tensor=inputs, output_tensor=outputs, momentum=_BATCH_NORM_DECAY,
      epsilon=_BATCH_NORM_EPSILON, center=True, scale=True, training=training)

  return outputs


def fixed_padding(inputs, kernel_size, data_format):
  """Pads the input along the spatial dimensions independently of input size.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 Should be a positive integer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    A tensor with the same format as the input with the data either intact
    (if kernel_size == 1) or padded (if kernel_size > 1).
  """
  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if data_format == 'channels_first':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]])
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
  return padded_inputs


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format, training):
  """Strided 2-D convolution with explicit padding."""
  # The padding is consistent and is based only on `kernel_size`, not on the
  # dimensions of `inputs` (as opposed to using `tf.layers.conv2d` alone).

  inputs_for_logging = inputs
  if strides > 1:
    inputs = fixed_padding(inputs, kernel_size, data_format)

  outputs = tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
      padding=('SAME' if strides == 1 else 'VALID'), use_bias=False,
      kernel_initializer=tf.variance_scaling_initializer(
          distribution="truncated_normal"),
      data_format=data_format, trainable=training)

  resnet_log_helper.log_conv2d(
      input_tensor=inputs_for_logging, output_tensor=outputs, stride=strides,
      filters=filters, initializer=mlperf_log.TRUNCATED_NORMAL, use_bias=False)

  return outputs


################################################################################
# ResNet block definitions.
################################################################################
def _building_block_v1(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  raise NotImplementedError


def _building_block_v2(inputs, filters, training, projection_shortcut, strides,
                       data_format):
  raise NotImplementedError


def _bottleneck_block_v1(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  """A single block for ResNet v1, with a bottleneck.

  Similar to _building_block_v1(), except using the "bottleneck" blocks
  described in:
    Convolution then batch normalization then ReLU as described by:
      Deep Residual Learning for Image Recognition
      https://arxiv.org/pdf/1512.03385.pdf
      by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the convolutions.
    training: A Boolean for whether the model is in training or inference
      mode. Needed for batch normalization.
    projection_shortcut: The function to use for projection shortcuts
      (typically a 1x1 convolution when downsampling the input).
    strides: The block's stride. If greater than 1, this block will ultimately
      downsample the input.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block; shape should match inputs.
  """
  resnet_log_helper.log_begin_block(
      input_tensor=inputs, block_type=mlperf_log.BOTTLENECK_BLOCK)

  shortcut = inputs

  if projection_shortcut is not None:
    shortcut = projection_shortcut(inputs)
    resnet_log_helper.log_projection(input_tensor=inputs,
                                     output_tensor=shortcut)
    shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format, training=training)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format, training=training)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
      data_format=data_format, training=training)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_SHORTCUT_ADD)
  inputs += shortcut

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  resnet_log_helper.log_end_block(output_tensor=inputs)
  return inputs


def _bottleneck_block_v2(inputs, filters, training, projection_shortcut,
                         strides, data_format):
  raise NotImplementedError


def block_layer(inputs, filters, bottleneck, block_fn, blocks, strides,
                training, name, data_format, version=1):
  """Creates one layer of blocks for the ResNet model.

  Args:
    inputs: A tensor of size [batch, channels, height_in, width_in] or
      [batch, height_in, width_in, channels] depending on data_format.
    filters: The number of filters for the first convolution of the layer.
    bottleneck: Is the block created a bottleneck block.
    block_fn: The block to use within the model, either `building_block` or
      `bottleneck_block`.
    blocks: The number of blocks contained in the layer.
    strides: The stride to use for the first convolution of the layer. If
      greater than 1, this layer will ultimately downsample the input.
    training: Either True or False, whether we are currently training the
      model. Needed for batch norm.
    name: A string name for the tensor output of the block layer.
    data_format: The input format ('channels_last' or 'channels_first').

  Returns:
    The output tensor of the block layer.
  """

  # Bottleneck blocks end with 4x the number of filters as they start with
  filters_out = filters * 4 if bottleneck else filters

  def projection_shortcut(inputs):
    return conv2d_fixed_padding(
        inputs=inputs, filters=filters_out, kernel_size=1, strides=strides,
        data_format=data_format, training=training)

  def projection_shortcut_v1d(inputs):
    avg_pool = tf.layers.average_pooling2d(
        inputs=inputs, pool_size=strides, strides=strides, padding='SAME',
        data_format=data_format)
    return conv2d_fixed_padding(
        inputs=avg_pool, filters=filters_out, kernel_size=1, strides=1,
        data_format=data_format, training=training)

  if version == 1:
    projection_shortcut = projection_shortcut
  elif version == 14:
    projection_shortcut = projection_shortcut_v1d

  # Only the first block per block_layer uses projection_shortcut and strides
  inputs = block_fn(inputs, filters, training, projection_shortcut, strides,
                    data_format)

  for _ in range(1, blocks):
    inputs = block_fn(inputs, filters, training, None, 1, data_format)

  return tf.identity(inputs, name)


class ModelTeach(object):
  """Base class for building the Resnet Model."""

  def __init__(self, resnet_size, bottleneck, num_classes, pickle_model,
               num_filters, kernel_size,
               conv_stride, first_pool_size, first_pool_stride,
               second_pool_size, second_pool_stride, block_sizes, block_strides,
               final_size, version=DEFAULT_VERSION, data_format=None,
               dtype=DEFAULT_DTYPE, temp_dst=2., w_dst=2., enable_at=False, w_at=2.):
    """Creates a model for classifying an image.

    Args:
      resnet_size: A single integer for the size of the ResNet model.
      bottleneck: Use regular blocks or bottleneck blocks.
      num_classes: The number of classes used as labels.
      num_filters: The number of filters to use for the first block layer
        of the model. This number is then doubled for each subsequent block
        layer.
      kernel_size: The kernel size to use for convolution.
      conv_stride: stride size for the initial convolutional layer
      first_pool_size: Pool size to be used for the first pooling layer.
        If none, the first pooling layer is skipped.
      first_pool_stride: stride size for the first pooling layer. Not used
        if first_pool_size is None.
      second_pool_size: Pool size to be used for the second pooling layer.
      second_pool_stride: stride size for the final pooling layer
      block_sizes: A list containing n values, where n is the number of sets of
        block layers desired. Each value should be the number of blocks in the
        i-th set.
      block_strides: List of integers representing the desired stride size for
        each of the sets of block layers. Should be same length as block_sizes.
      final_size: The expected size of the model after the second pooling.
      version: Integer representing which version of the ResNet network to use.
        See README for details. Valid values: [1, 2]
      data_format: Input format ('channels_last', 'channels_first', or None).
        If set to None, the format is dependent on whether a GPU is available.
      dtype: The TensorFlow dtype to use for calculations. If not specified
        tf.float32 is used.

    Raises:
      ValueError: if invalid version is selected.
    """
    self.resnet_size = resnet_size

    if not data_format:
      data_format = (
          'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    self.resnet_version = version
    if version not in (1, 2, 14):
      raise ValueError(
          'Resnet version should be 1 or 2. See README for citations.')

    self.bottleneck = bottleneck
    if bottleneck:
      if (version == 1) or (version == 14):
        self.block_fn = _bottleneck_block_v1
      else:
        self.block_fn = _bottleneck_block_v2
    else:
      if (version == 1) or (version == 14):
        self.block_fn = _building_block_v1
      else:
        self.block_fn = _building_block_v2

    if dtype not in ALLOWED_TYPES:
      raise ValueError('dtype must be one of: {}'.format(ALLOWED_TYPES))

    save_pkl = pickle_model #'./gap_save/gap_pruned.pkl'
    if save_pkl.startswith("oss://"):
      with tf.gfile.GFile(save_pkl, "r") as f:
        gap_vars = pickle.load(f)
    else:
      with open(save_pkl, 'rb') as f:
        gap_vars = pickle.load(f)

    self.gf = gap_finetune.GapFinetune(gap_vars)

    self.temp_dst = temp_dst
    self.w_dst = w_dst
    self.enable_at = enable_at
    self.w_at = w_at # 4 for per-batch 64, 8 for 128

    self.data_format = data_format
    self.num_classes = num_classes
    self.num_filters = num_filters
    self.kernel_size = kernel_size
    self.conv_stride = conv_stride
    self.first_pool_size = first_pool_size
    self.first_pool_stride = first_pool_stride
    self.second_pool_size = second_pool_size
    self.second_pool_stride = second_pool_stride
    self.block_sizes = block_sizes
    self.block_strides = block_strides
    self.final_size = final_size
    self.dtype = dtype
    self.pre_activation = version == 2

  def _custom_dtype_getter(self, getter, name, shape=None, dtype=DEFAULT_DTYPE,
                           *args, **kwargs):
    """Creates variables in fp32, then casts to fp16 if necessary.

    This function is a custom getter. A custom getter is a function with the
    same signature as tf.get_variable, except it has an additional getter
    parameter. Custom getters can be passed as the `custom_getter` parameter of
    tf.variable_scope. Then, tf.get_variable will call the custom getter,
    instead of directly getting a variable itself. This can be used to change
    the types of variables that are retrieved with tf.get_variable.
    The `getter` parameter is the underlying variable getter, that would have
    been called if no custom getter was used. Custom getters typically get a
    variable with `getter`, then modify it in some way.

    This custom getter will create an fp32 variable. If a low precision
    (e.g. float16) variable was requested it will then cast the variable to the
    requested dtype. The reason we do not directly create variables in low
    precision dtypes is that applying small gradients to such variables may
    cause the variable not to change.

    Args:
      getter: The underlying variable getter, that has the same signature as
        tf.get_variable and returns a variable.
      name: The name of the variable to get.
      shape: The shape of the variable to get.
      dtype: The dtype of the variable to get. Note that if this is a low
        precision dtype, the variable will be created as a tf.float32 variable,
        then cast to the appropriate dtype
      *args: Additional arguments to pass unmodified to getter.
      **kwargs: Additional keyword arguments to pass unmodified to getter.

    Returns:
      A variable which is cast to fp16 if necessary.
    """

    if dtype in CASTABLE_TYPES:
      var = getter(name, shape, tf.float32, *args, **kwargs)
      return tf.cast(var, dtype=dtype, name=name + '_cast')
    else:
      return getter(name, shape, dtype, *args, **kwargs)

  def _model_variable_scope(self):
    """Returns a variable scope that the model should be created under.

    If self.dtype is a castable type, model variable will be created in fp32
    then cast to self.dtype before being used.

    Returns:
      A variable scope for the model.
    """

    return tf.variable_scope('resnet_model_teach',
                             custom_getter=self.gf.get_variable)


  def calc_loss_dst(self, logits, logits_dst):
    labels_soft = tf.nn.softmax(logits_dst / self.temp_dst)
    loss = self.w_dst * tf.losses.softmax_cross_entropy(labels_soft, logits)
    tf.summary.scalar('dist_loss', loss)
    return loss


  def calc_loss_at(self, feat_t, feat_s):
    at_loss = 0.
    for x_t, x_s in zip(feat_t, feat_s):
      if self.data_format == 'channels_first':
        reduce_axis = 1
      elif self.data_format == 'channels_last':
        reduce_axis = -1
      vec_t = tf.layers.flatten(tf.reduce_mean(tf.pow(x_t, 2), axis=reduce_axis))
      vec_s = tf.layers.flatten(tf.reduce_mean(tf.pow(x_s, 2), axis=reduce_axis))
      norm_t = tf.nn.l2_normalize(vec_t, axis=1)
      norm_s = tf.nn.l2_normalize(vec_s, axis=1)
      at_loss += tf.reduce_mean(tf.norm(norm_t-norm_s, ord=2, axis=1), axis=0)
    at_loss = self.w_at * at_loss
    tf.summary.scalar('at_loss', at_loss)
    return at_loss


  def __call__(self, inputs, training):
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, self.num_classes].
    """

    # Drop batch size from shape logging.
    mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_INITIAL_SHAPE,
                            value=inputs.shape.as_list()[1:])

    with self._model_variable_scope():
      if self.data_format == 'channels_first':
        # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
        # This provides a large performance boost on GPU. See
        # https://www.tensorflow.org/performance/performance_guide#data_formats
        inputs = tf.transpose(inputs, [0, 3, 1, 2])

      if (self.resnet_version == 1) or (self.resnet_version == 2):
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=self.kernel_size,
            strides=self.conv_stride, data_format=self.data_format, training=training)

      elif self.resnet_version == 14: # v1d architecture
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=3,
            strides=self.conv_stride, data_format=self.data_format, training=training)
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=3,
            strides=1, data_format=self.data_format, training=training)
        inputs = batch_norm(inputs, training, self.data_format)
        inputs = conv2d_fixed_padding(
            inputs=inputs, filters=self.num_filters, kernel_size=3,
            strides=1, data_format=self.data_format, training=training)

      # We do not include batch normalization or activation functions in V2
      # for the initial conv1 because the first ResNet unit will perform these
      # for both the shortcut and non-shortcut paths as part of the first
      # block's projection. Cf. Appendix of [2].
      if (self.resnet_version == 1) or (self.resnet_version == 14):
        inputs = batch_norm(inputs, training, self.data_format)

        mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
        inputs = tf.nn.relu(inputs)

      if self.first_pool_size:
        pooled_inputs = tf.layers.max_pooling2d(
            inputs=inputs, pool_size=self.first_pool_size,
            strides=self.first_pool_stride, padding='SAME',
            data_format=self.data_format)
        resnet_log_helper.log_max_pool(input_tensor=inputs, output_tensor=pooled_inputs)
        inputs = tf.identity(pooled_inputs, 'initial_max_pool')

      feat_t = list()
      for i, num_blocks in enumerate(self.block_sizes):
        num_filters = self.num_filters * (2**i)
        inputs = block_layer(
            inputs=inputs, filters=num_filters, bottleneck=self.bottleneck,
            block_fn=self.block_fn, blocks=num_blocks,
            strides=self.block_strides[i], training=training,
            name='block_layer{}'.format(i + 1), data_format=self.data_format,
            version=self.resnet_version)
        if (i > 1) and self.enable_at:
          feat_t.append(inputs)

      # Only apply the BN and ReLU for model that does pre_activation in each
      # building/bottleneck block, eg resnet V2.
      if self.pre_activation:
        inputs = batch_norm(inputs, training, self.data_format)

        mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
        inputs = tf.nn.relu(inputs)

      # The current top layer has shape
      # `batch_size x pool_size x pool_size x final_size`.
      # ResNet does an Average Pooling layer over pool_size,
      # but that is the same as doing a reduce_mean. We do a reduce_mean
      # here because it performs better than AveragePooling2D.
      axes = [2, 3] if self.data_format == 'channels_first' else [1, 2]
      inputs = tf.reduce_mean(inputs, axes, keepdims=True)
      inputs = tf.identity(inputs, 'final_reduce_mean')

      inputs = tf.reshape(inputs, [-1, inputs.get_shape().as_list()[-1]])
      mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_DENSE,
                              value=self.num_classes)
      inputs = tf.layers.dense(
        inputs=inputs,
        units=self.num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=.01),
        trainable=training)
      inputs = tf.identity(inputs, 'final_dense')

      # Drop batch size from shape logging.
      mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_FINAL_SHAPE,
                              value=inputs.shape.as_list()[1:])
      return inputs, feat_t