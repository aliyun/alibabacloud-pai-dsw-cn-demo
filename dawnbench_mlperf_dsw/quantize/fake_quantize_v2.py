
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
"""Logic to update a TensorFlow model graph with quantization operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import pdb
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging

from quantize import common
from quantize import input_to_ops

# Quantizable operation types that are supported by the quantization rewrite.
_QUANTIZABLE_TYPES = {'Conv2D', 'MatMul', 'DepthwiseConv2dNative'}
_IGNORE_REROUTE = {'Reshape', 'Identity', 'Pad', 'Add', 'L2Loss', 'AvgPool'}

_ALL_QUANTIZE_BITS = [4, 8, 16, 32]
_KL_SCOPE = 'kl_quantize'
_GRADIENTS_SCOPE = 'gradients'
_AP_SCOPE = 'tower'

def quantize_graph(graph=None,
                   bits=8,
                   per_channel=True,
                   include_scopes=None,
                   exclude_scopes=['resnet_model_teach'],
                   model_scope='resnet_model',
                   pre_calib='/app/calib/',
                   int8_layers=[],
                   quant_copy_num=4,
                   online=False,
                   method=1):
  assert bits in [4, 8, 16, 32], \
      ('Expected int4, int8, float16 or float32 for quantization bits, \
        was set %d for %s ' % (num_bits, inputs.name))

  if bits == 32:
    logging.info("Graph remained unchanged as the quantization mode is set to float32")
    return

  g_rewriter = GraphRewriterAMP(graph,
                                bits=bits,
                                per_channel=per_channel,
                                include_scopes=include_scopes,
                                exclude_scopes=exclude_scopes,
                                model_scope=model_scope,
                                pre_calib=pre_calib,
                                int8_layers=int8_layers,
                                quant_copy_num=quant_copy_num,
                                online=online,
                                method=method)
  return g_rewriter._act_dict

class GraphRewriterAMP(object):
  def __init__(self,
               graph=None,
               bits=8,
               per_channel=True,
               include_scopes=None,
               exclude_scopes=None,
               model_scope='model',
               pre_calib='',
               int8_layers=[],
               quant_copy_num=4,
               online=False,
               method=1,
               target_ops=None,
               reuse=None):
    if not graph:
      self._graph = ops.get_default_graph()
    else:
      self._graph = graph

    self._bits = bits
    self._per_channel = per_channel
    self._model_scope = model_scope
    self._int8_layers = int8_layers
    self._copy_num = quant_copy_num
    self._online = online
    self._method = method

    self._pre_calib = os.path.join(pre_calib, 'act_dict_%s.npy' % self._bits)
    if self._method == 2:
      print('Activation is per-tensor quantized by MAX method')
      self._act_dict = dict()
    else:
      if not os.path.isfile(self._pre_calib):
        print('No pre-calib KL scale is offered, which should be calculated firstly.')
        self._act_dict = dict()
      else:
        print('Load the pre-calib: %s' % self._pre_calib)
        self._act_dict = np.load(self._pre_calib, allow_pickle=True).item()

    if include_scopes:
      if type(include_scopes) is not list:
        include_scopes = [include_scopes]
      for i, scope in enumerate(include_scopes):
        if not include_scopes[i].endwith('/'):
          include_scopes[i] += '/'
    self._include_scopes = include_scopes
    self._target_ops = target_ops
    self._reuse = reuse

    if not exclude_scopes:
      exclude_scopes = [_GRADIENTS_SCOPE]
    if type(exclude_scopes) is not list:
      exclude_scopes = [exclude_scopes]
    exclude_scopes +=  [_GRADIENTS_SCOPE]
    for i, scope in enumerate(exclude_scopes):
      if not exclude_scopes[i].endswith('/'):
        exclude_scopes[i] += '/'
    self._exclude_scopes = exclude_scopes

    with self._graph.as_default():
      self._quantize(self._bits)


  def _find_quantize_ops(self, graph, target_ops=None, include_scopes=None, exclude_scopes=None):
    if target_ops is not None:
      candidate_ops = common.TraceBackOps(target_ops)
    else:
      candidate_ops = graph.get_operations()

    to_quantize_ops = [op for op in candidate_ops \
        if op.type in _QUANTIZABLE_TYPES]

    if include_scopes:
      if type(include_scopes) is not list:
        include_scopes = [include_scopes]
      to_quantize_ops_selected = []
      for op in to_quantize_ops:
        for scope in include_scopes:
          if op.name.startswith(scope):
            to_quantize_ops_selected.append(op)
            break
      to_quantize_ops = to_quantize_ops_selected

    if exclude_scopes:
      if type(exclude_scopes) is not list:
        exclude_scopes = [exclude_scopes]
      to_quantize_ops_selected = []
      for op in to_quantize_ops:
        exclude = False
        for scope in exclude_scopes:
          if op.name.startswith(scope):
            exclude = True
            break
        if not exclude:
          to_quantize_ops_selected.append(op)
      to_quantize_ops = to_quantize_ops_selected

    return to_quantize_ops

  def _quantize(self, bits=8):
    input_to_ops_map = input_to_ops.InputToOps(self._graph)
    quantized_ops = set()
    to_quantize_ops = self._find_quantize_ops(
        self._graph,
        self._target_ops,
        self._include_scopes,
        self._exclude_scopes)

    for i, to_quantize_op in enumerate(to_quantize_ops):
      context = _GetContextFromOp(to_quantize_op)
      weight_op = to_quantize_op.inputs[1].op
      input_op = to_quantize_op.inputs[0].op

      weight_tensor = to_quantize_op.inputs[1]
      input_tensor = to_quantize_op.inputs[0]
      # ignore the teacher in KD, and backward nodes
      if (self._model_scope+'_teach' in context) \
         or (_GRADIENTS_SCOPE in context):
        continue

      # control the tower copies when data parallel training
      tower_id = 0
      if _AP_SCOPE in context:
        tower_id = int(re.findall(r"\d+", context.split('/')[0])[0])
        if tower_id >= self._copy_num:
          continue

      #if 'resnet_model/Pad:0' in input_tensor.name:
        #  print(input_op.inputs[0].name)
      # set the num_bits=8 for int8 layers
      layer_bits = bits
      for int8_layer in self._int8_layers:
        if int8_layer in to_quantize_op.name:
          layer_bits = 8
          break

      with tf.device('/gpu:%s' % tower_id):
          # Quantize the weights.
          #if not self._online:
          self._insert_quant_op(
              context,
              'weights_quant',
              weight_op,
              input_to_ops_map.ConsumerOperations(weight_op),
              bits_options=layer_bits,
              per_channel=self._per_channel,
              consumer_scope=self._include_scopes,
              reuse=self._reuse,
              is_activation=False,
              is_depthwise=to_quantize_op.type=="DepthwiseConv2dNative")

          # Quantize the input.
          if not (_KL_SCOPE in input_tensor.name):
            self._insert_quant_op(
              context,
              'input_quant',
              input_op,
              input_to_ops_map.ConsumerOperations(input_op),
              bits_options=layer_bits,
              per_channel=False,
              consumer_scope=self._include_scopes,
              reuse=self._reuse,
              is_activation=True,
              is_depthwise=to_quantize_op.type=="DepthwiseConv2dNative")


  def _insert_quant_op(self,
                       context,
                       name,
                       producer,
                       consumers,
                       bits_options=None,
                       per_channel=False,
                       consumer_scope=None,
                       reuse=None,
                       is_activation=False,
                       is_depthwise=False):

    if consumer_scope:
      consumers_in_scope = []
      for consumer in consumers:
        if consumer.name.startswith(consumer_scope):
          consumers_in_scope.append(consumer)
        else:
          logging.info(
              '_InsertQuantOp context="%s" name="%s" ignores '
              'consumer "%s" because it is not in scope "%s"',
              context, name, consumer.name, consumer_scope)
          return
      consumers = consumers_in_scope

    name_prefix = _AddContextToName(context, name)

    name_scope = self._graph.get_name_scope()
    if name_scope:
      name_prefix = common.DropStringPrefix(name_prefix, name_scope + '/')

    inputs = producer.outputs[0]

    if bits_options is None:
      bits_options = _ALL_QUANTIZE_BITS

    if type(bits_options) is not list:
      num_bits = bits_options
    elif len(bits_options) == 1:
      num_bits = bits_options[0]

    quant = self._online_quantize(inputs,
                                  num_bits=num_bits,
                                  per_channel=per_channel,
                                  name_prefix=name_prefix,
                                  is_activation=is_activation,
                                  is_depthwise=is_depthwise)

    if consumers:
      tensors_modified_count = common.RerouteTensor(
          quant, inputs, can_modify=consumers, ignored_ops=_IGNORE_REROUTE)


  def _online_quantize(self,
                       inputs,
                       num_bits=8,
                       per_channel=True,
                       name_prefix=None,
                       is_activation=False,
                       is_depthwise=False):
    if _AP_SCOPE in name_prefix:
      kl_scope = name_prefix.split('/')[0]+'/'+_KL_SCOPE
    else:
      kl_scope = _KL_SCOPE

    with tf.name_scope(kl_scope) as scope:
        inputs = self._online_quantize_impl(
                                          inputs,
                                          num_bits=num_bits,
                                          per_channel=per_channel,
                                          is_activation=is_activation,
                                          is_depthwise=is_depthwise)
        return inputs

  def _online_quantize_impl(self,
                            inputs,
                            num_bits=8,
                            per_channel=True,
                            is_activation=False,
                            is_depthwise=False):
    input_name = inputs.name.split(':')[0]
    input_shape = inputs.shape.as_list()
    if is_activation:
      if self._method == 2: # MAX
        if len(input_shape) == 4:
          max_x = tf.reduce_max(tf.abs(inputs), [0, 1, 2, 3])
        elif len(input_shape) == 2:
          max_x = tf.reduce_max(tf.abs(inputs), [0, 1])
        scale_t = tf.divide((pow(2, num_bits) / 2 - 1), max_x)
      else:
        idx = input_name.index(self._model_scope)
        act_name = input_name[idx:] +':0'

        if act_name in self._act_dict:
          scale_value = self._act_dict[act_name][0]
        else:
          scale_value = 1.
          self._act_dict[act_name] = (1., num_bits)
        if self._online:
          scale_value = 1.
        scale_t = tf.constant(scale_value, dtype=tf.float32, name=input_name+'_const')
        #scale = scale_value
        #print(scale.name, num_bits, scale_value)
    else:
      if per_channel:
        if is_depthwise:
          max_x = tf.reshape(tf.reduce_max(tf.abs(inputs), [0, 1, 3]), [1, 1, input_shape[2], 1])
        else:
          if len(input_shape) == 4:
            max_x = tf.reshape(tf.reduce_max(tf.abs(inputs), [0, 1, 2]), [1, 1, 1, input_shape[-1]])
          elif len(input_shape) == 2:
            max_x = tf.reshape(tf.reduce_max(tf.abs(inputs), [0,]), [1, input_shape[-1]])

      #print('Weights: [{:}], bits: {:}'.format(input_name, num_bits))
      scale_t = tf.divide((pow(2, num_bits) / 2 - 1), max_x)

    scale = tf.stop_gradient(scale_t)
    min_value, max_value = float(-1 * (pow(2, num_bits) / 2)), float(pow(2, num_bits) / 2 - 1)

    with self._graph.gradient_override_map({'Round': 'Identity'}):
      quant_x = tf.round(scale * inputs)
      if is_activation:
          quant_x = tf.clip_by_value(quant_x, min_value, max_value)
      quant_x = tf.divide(quant_x, scale)

    inputs_quant = quant_x
    inputs_quant.set_shape(inputs.shape)
    return inputs_quant


def _GetContextFromOp(op):
  """Gets the root context name from the op name."""
  context_re = re.search(r'^(.*)/([^/]+)', op.name)
  if context_re:
    return context_re.group(1)
  return ''


def _AddContextToName(context, name):
  """Adds the context to the name if it exists."""
  if not context:
    return name
  return context + '/' + name

def _ModelVariable(name,
                   shape=None,
                   initializer=None,
                   collections=None,
                   trainable=None):
  if type(collections) is str:
    collections = [collections]
  collections = list(collections or [])
  collections += [ops.GraphKeys.GLOBAL_VARIABLES]
  return variable_scope.get_variable(
      name,
      shape=shape,
      initializer=initializer,
      collections=collections,
      trainable=trainable,
      aggregation=variable_scope.VariableAggregation.MEAN)
