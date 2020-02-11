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
"""Common utilities used across this package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import pdb

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope

# Skip all operations that are backprop related or export summaries.
SKIPPED_PREFIXES = (
    'Const_', 'HistogramSummary',
    'ScalarSummary')

# Valid activation ops for quantization end points.
_ACTIVATION_OP_SUFFIXES = ['Relu6', 'Relu', 'Identity']

# Regular expression for recognizing nodes that are part of batch norm group.
_BATCHNORM_RE = re.compile(r'^(.*)BatchNorm/batchnorm')


def BatchNormGroups(graph):
  """Finds batch norm layers, returns their prefixes as a list of strings.

  Args:
    graph: Graph to inspect.

  Returns:
    List of strings, prefixes of batch norm group names found.
  """
  bns = []
  for op in graph.get_operations():
    match = _BATCHNORM_RE.search(op.name)
    if match:
      bn = match.group(1)
      if not bn.startswith(SKIPPED_PREFIXES):
        bns.append(bn)
  # Filter out duplicates.
  return list(collections.OrderedDict.fromkeys(bns))


def CreateOrGetQuantizationStep():
  """Returns a Tensor of the number of steps the quantized graph has run.

  Returns:
    Quantization step Tensor.
  """
  quantization_step_name = 'fake_quantization_step'
  quantization_step_tensor_name = quantization_step_name + '/Identity:0'
  g = ops.get_default_graph()
  try:
    return g.get_tensor_by_name(quantization_step_tensor_name)
  except KeyError:
    # Create in proper graph and base name_scope.
    with g.name_scope(None):
      quantization_step_tensor = variable_scope.get_variable(
          quantization_step_name,
          shape=[],
          dtype=dtypes.int64,
          initializer=init_ops.zeros_initializer(),
          trainable=False,
          collections=[ops.GraphKeys.GLOBAL_VARIABLES],
          aggregation=variable_scope.VariableAggregation.ONLY_FIRST_REPLICA)
      with g.name_scope(quantization_step_tensor.op.name + '/'):
        # We return the incremented variable tensor. Since this is used in conds
        # for quant_delay and freeze_bn_delay, it will run once per graph
        # execution. We return an identity to force resource variables and
        # normal variables to return a tensor of the same name.
        return array_ops.identity(
            state_ops.assign_add(quantization_step_tensor, 1))


def DropStringPrefix(s, prefix):
  """If the string starts with this prefix, drops it."""
  if s.startswith(prefix):
    return s[len(prefix):]
  else:
    return s


def RerouteTensor(t0, t1, can_modify=None, ignored_ops=[]):
  """Reroute the end of the tensor t0 to the ends of the tensor t1.

  Args:
    t0: a tf.Tensor.
    t1: a tf.Tensor.
    can_modify: iterable of operations which can be modified. Any operation
      outside within_ops will be left untouched by this function.

  Returns:
    The number of individual modifications made by the function.
  """
  nb_update_inputs = 0
  consumers = t1.consumers()
  if can_modify is not None:
    consumers = [c for c in consumers if c in can_modify]
  for c in consumers:
    if c.type in ignored_ops:
      #print("No input update for: %s" % c.name)
      consumers.remove(c)
  consumers_indices = {}
  for c in consumers:
    consumers_indices[c] = [i for i, t in enumerate(c.inputs) if t is t1]
  for c in consumers:
    #print("Input update for: %s" % c.name)
    for i in consumers_indices[c]:
      c._update_input(i, t0)  # pylint: disable=protected-access
      nb_update_inputs += 1
  return nb_update_inputs

def TraceBackOps(target_ops=None):
  if type(target_ops) is not list:
    target_ops = [target_ops]
  trace_back_ops = target_ops
  for i, target in enumerate(trace_back_ops):
    if type(trace_back_ops[i]) is ops.Tensor:
      trace_back_ops[i] = trace_back_ops[i].op
  num_ops = len(trace_back_ops)
  while True:
    for op in trace_back_ops[:]:
      for input in op.inputs:
        if input.op not in trace_back_ops:
          trace_back_ops.append(input.op)
    new_num_ops = len(trace_back_ops)
    if new_num_ops == num_ops:
      break

    num_ops = new_num_ops
  return trace_back_ops
