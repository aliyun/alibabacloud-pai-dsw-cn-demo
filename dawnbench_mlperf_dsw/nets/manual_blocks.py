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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from mlperf_compliance import mlperf_log
from mlperf_compliance import resnet_log_helper


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
      scale=True, training=training, fused=True)

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


def conv2d_fixed_padding(inputs, filters, kernel_size, strides, data_format):
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
      data_format=data_format)

  resnet_log_helper.log_conv2d(
      input_tensor=inputs_for_logging, output_tensor=outputs, stride=strides,
      filters=filters, initializer=mlperf_log.TRUNCATED_NORMAL, use_bias=False)

  return outputs


################################################################################
# ResNet block definitions.
################################################################################
def block_m1(inputs, filters, training, strides, data_format):

  resnet_log_helper.log_begin_block(
      input_tensor=inputs, block_type=mlperf_log.BOTTLENECK_BLOCK)

  avg_pool = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=strides, strides=strides, padding='SAME',
      data_format=data_format)
  shortcut = conv2d_fixed_padding(
        inputs=avg_pool, filters=640, kernel_size=1, strides=1,
        data_format=data_format)
  resnet_log_helper.log_projection(input_tensor=inputs,
                                     output_tensor=shortcut)
  shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=640, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_SHORTCUT_ADD)
  inputs += shortcut

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  resnet_log_helper.log_end_block(output_tensor=inputs)
  return inputs


def block_m2(inputs, filters, training, strides, data_format):

  resnet_log_helper.log_begin_block(
      input_tensor=inputs, block_type=mlperf_log.BOTTLENECK_BLOCK)

  shortcut = inputs

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=filters, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=640, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_SHORTCUT_ADD)
  inputs += shortcut

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  resnet_log_helper.log_end_block(output_tensor=inputs)
  return inputs


def block_m3(inputs, filters, training, strides, data_format):

  resnet_log_helper.log_begin_block(
      input_tensor=inputs, block_type=mlperf_log.BOTTLENECK_BLOCK)

  avg_pool = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=strides, strides=strides, padding='SAME',
      data_format=data_format)
  shortcut = conv2d_fixed_padding(
        inputs=avg_pool, filters=1664, kernel_size=1, strides=1,
        data_format=data_format)
  resnet_log_helper.log_projection(input_tensor=inputs,
                                     output_tensor=shortcut)
  shortcut = batch_norm(inputs=shortcut, training=training,
                          data_format=data_format)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=256, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=256, kernel_size=3, strides=strides,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=1664, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_SHORTCUT_ADD)
  inputs += shortcut

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  resnet_log_helper.log_end_block(output_tensor=inputs)
  return inputs


def block_m4(inputs, filters, training, strides, data_format):

  resnet_log_helper.log_begin_block(
      input_tensor=inputs, block_type=mlperf_log.BOTTLENECK_BLOCK)

  shortcut = inputs

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=384, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=384, kernel_size=3, strides=strides, # 384
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  inputs = conv2d_fixed_padding(
      inputs=inputs, filters=1664, kernel_size=1, strides=1,
      data_format=data_format)
  inputs = batch_norm(inputs, training, data_format)

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_SHORTCUT_ADD)
  inputs += shortcut

  mlperf_log.resnet_print(key=mlperf_log.MODEL_HP_RELU)
  inputs = tf.nn.relu(inputs)

  resnet_log_helper.log_end_block(output_tensor=inputs)
  return inputs
