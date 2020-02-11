"""
  Code for GAP tools: graph comprehension
  Usage document ref to https://lark.alipay.com/pai/developer-docs/gap_tools
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pickle
import re
import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import gfile
from tensorflow.python.training import saver

from tensorflow.contrib.nccl.python.ops import nccl_ops
nccl_ops._maybe_load_nccl_ops_so()

_BASE_LEN = 32

_CHANNEL_AXIS = [3, -1]

_VARIABLE_OPS = [\
  "Assign",
  "AssignAdd",
  "AssignSub",
  "Queue",
  "ScatterAdd",
  "ScatterSub",
  "ScatterUpdate",
  "TruncatedNormal",
  "Variable",
  "VariableV2"]

_KEY_OPS = [\
  'Conv2D',
  'DepthwiseConv2dNative',
  'SpaceToBatchND',
  'BatchToSpaceND',
  'BN',
  'Relu',
  'ConcatV2',
  'Split',
  'SplitV',
  'AvgPool',
  'MaxPool',
  'BiasAdd',
  'MatMul',
  #'Squeeze',
  'ResizeBilinear']

_KEY_OPS_NONCHANNEL = [\
  'Relu',
  'AvgPool',
  'MaxPool',
  'BiasAdd',
  'DepthwiseConv2dNative',
  'SpaceToBatchND',
  'BatchToSpaceND',
  'Squeeze',
  'Max',
  'Min',
  'Sum',
  'ResizeBilinear']

_KEY_OPS_SINGLE_OUTPUT = [\
  'Conv2D',
  'DepthwiseConv2dNative',
  'SpaceToBatchND',
  'BatchToSpaceND',
  #'Relu',
  'ConcatV2',
  'AvgPool',
  'MaxPool',
  'BiasAdd',
  'MatMul',
  'ResizeBilinear']

_SPLIT_OPS = [\
  'Split',
  'SplitV']

_CONCAT_OPS = [\
  'ConcatV2']

_SHORTCUT_OPS = [\
  'Cast',
  'Identity'
]
_REDUCE_OPS = [\
  'Sum',
  'Max',
  'Min'
]

_OPTIMIZERS = [\
  'Adadelta', 'Adagrad',
  'AdamAsync', 'Adam',
  'Ftrl', 'GradientDescent',
  'Momentumm', 'WeightedMovingAvg',
  'RMSProp'
]

_SKIPPED_OPS = ['Shape', 'ShapeN']
_GRADIENTS_SCOPE = 'gradients'

def IsShortcutOp(op):
  if _GRADIENTS_SCOPE in op.name:
    return False
  #if op.type in _SHORTCUT_OPS:
  #  return True
  if op.type in _REDUCE_OPS:
    if op.inputs[0].shape.ndims is None:
      return False
    input_rank = op.inputs[0].shape.ndims
    axis = op.inputs[1]
    if axis.op.type == 'Const':
      tensor = axis.op.node_def.attr['value'].tensor
      tensor_dtype = dtypes.as_dtype(tensor.dtype)
      dtype = tensor_dtype.as_numpy_dtype
      if len(tensor.int_val) == 0:
        return False
      axis_value = list(np.fromiter(tensor.int_val, dtype=dtype))
      if -1 in axis_value:
        return False

      if all([v < input_rank-1 for v in axis_value]):
        return True

  return False

def IsOptimizerVars(var_name):
  for keyword in _OPTIMIZERS:
    if keyword in var_name:
      return True
  return False

def gap_get_l1_loss(graph, summ=False):
  """ Interface to get GAP l1 loss (step1).
    Args:
      graph: Grap for the to-be-processed graph
      summ: tf.summary or not
    Return:
      l1_loss: l1_loss tensor of scaling factors in batch normalization layers.
  """
  key_graph = KeyGraph(graph)
  l1_loss = key_graph.gap_l1_loss(summ)
  return l1_loss


def gap_perform_pruning(model_path, pruned_save_path=None, mode='gap', slim_ratio=0.5,
                        mask_len=False, full_save=False, full_save_path=None, var_scope='',
                        ver=1):
  """ Interface for GAP pruning step (step2).
    Args:
      model_path: path to the saved checkpoint,
         including 3 files: `.meta', `.data' and `.index'.
      pruned_save_path: path to save the pruned data (file in pickle format)
      slim_ratio: ratio for model pruning.
    Return:
      data_dict: the pruned data dict
  """
  graph = saver.import_meta_graph(model_path+'.meta', clear_devices=True)
  with open('graph_def.pbtxt', 'w') as f:
    f.write(str(ops.get_default_graph().as_graph_def(add_shapes=True)))
  key_graph = KeyGraph(ops.get_default_graph())
  data_dict = key_graph.gap(model_path, pruned_save_path, mode, slim_ratio, mask_len,
                            full_save, full_save_path, var_scope, ver)
  return data_dict


class KeyNode(object):
  """Key node structure."""
  def __init__(self,
               name,
               op_type,
               input_size=None,
               output_size=None,
               input_nodes=None,
               output_nodes=None,
               variables=None):
    self.name = name
    self.op_type = op_type
    self.input_size = []
    self.output_size = []
    self.input_nodes = []
    self.output_nodes = []
    self.variables = {}
    self.variables_name = {}
    self.input_gammas = []
    self.output_gammas = []

    self.add_input_nodes(input_nodes)
    self.add_output_nodes(output_nodes)
    self.add_variable(variables)

    if input_size is not None:
      self.set_input_size(input_size)
    if output_size is not None:
      self.set_output_size(output_size)

  def set_input_size(self, input_size):
    self.input_size = []
    if not isinstance(input_size, list):
      input_size = [input_size]
    self.input_size.extend(input_size)

  def set_output_size(self, output_size):
    self.output_size = []
    if not isinstance(output_size, list):
      output_size = [output_size]
    self.output_size.extend(output_size)

  def add_input_nodes(self, nodes):
    if nodes is None:
      return
    if not isinstance(nodes, list):
      nodes = [nodes]
    self.input_nodes.extend(nodes)
    return

  def add_output_nodes(self, nodes):
    if nodes is None:
      return
    if not isinstance(nodes, list):
      nodes = [nodes]
    self.output_nodes.extend(nodes)
    return

  def set_input_nodes(self):
    self.input_nodes = set(self.input_nodes)
    self.input_nodes = list(self.input_nodes)

  def set_output_nodes(self):
    self.output_nodes = set(self.output_nodes)
    self.output_nodes = list(self.output_nodes)

  def add_variable(self, var):
    if var is None:
      return

    for k, v in var.items():
      self.variables[k] = v
      if isinstance(v, ops.Operation):
        self.variables_name[k] = v.node_def.name
    return

  def add_input_gammas(self, gammas):
    if not isinstance(gammas, list):
      gammas = [gammas]
    self.input_gammas.extend(gammas)

  def add_output_gammas(self, gammas):
    if not isinstance(gammas, list):
      gammas = [gammas]
    self.output_gammas.extend(gammas)

class GammaStructure(object):
  """Gamma (scale factor in in BN layer) variable structure"""
  def __init__(self,
               name,
               gamma_mode='Plain',
               start_inx=0,
               end_inx=-1,
               shape=-1):

    self.name = name
    # gamma_mode: Plain, Split, Concat, AllSelect
    self.gamma_mode = gamma_mode
    self.start_inx = start_inx
    self.end_inx = end_inx
    self.shape = shape
    self.corr_gammas = list()

  def set_start_inx(self, start_inx):
    self.start_inx = start_inx

  def set_end_inx(self, end_inx):
    self.end_inx = end_inx

  def set_gamma_mode(self, gamma_mode):
    self.gamma_mode = gamma_mode

  def set_indices(self, indices):
    # for `Concat' mode
    self.indices = copy.deepcopy(indices)

  def add_corr_gammas(self, gammas):
    if not isinstance(gammas, list):
      gammas = [gammas]
    for g in gammas:
      if g not in self.corr_gammas:
        self.corr_gammas.append(g)

  def get_value(self, data_dict):
    """Function for getting the value for current gamma structure, given data dict"""
    value = []
    if self.gamma_mode == 'AllSelect':
      value = np.ones(self.shape, dtype=np.bool)
    elif self.gamma_mode == 'Plain':
      value = data_dict[self.name]
    elif self.gamma_mode == 'Split':
      value = self.corr_gammas[0].get_value(data_dict)
      value = value[self.start_inx:self.end_inx]
    elif self.gamma_mode == 'Concat':
      for sub_gamma in self.corr_gammas:
        sub_value = sub_gamma.get_value(data_dict)
        value.extend(sub_value)
    return list(value)

  def update_value(self, data_dict, update_value):
    """Function for updating the value for current gamma structure, given data dict and the update value"""
    if self.gamma_mode == 'Plain':
      data_dict[self.name] = update_value
    elif self.gamma_mode == 'Split':
      value = self.corr_gammas[0].get_value(data_dict)
      value[self.start_inx:self.end_inx] = update_value
      self.corr_gammas[0].update_value(data_dict, value)
    elif self.gamma_mode == 'Concat':
      i = 0
      for sub_gamma in self.corr_gammas:
        sub_gamma.update_value(\
            data_dict,
            update_data[self.indices[i]:self.indices[i+1]])
        i += 1


class BNStructure(object):
  """Structure that records all the variables in a single BN layer"""
  def __init__(self, name):
    self.name = name # keyed by gamma name
    self.variables = {}
    self.mask = []

  def add_variable(self,
                   gamma=None,
                   beta=None,
                   moving_mean=None,
                   moving_variance=None):
    """Add BN related variables to the variable dict"""
    if gamma is not None:
      self.variables['gamma'] = gamma
    if beta is not None:
      self.variables['beta'] = beta
    if moving_mean is not None:
      self.variables['moving_mean'] = moving_mean
    if moving_variance is not None:
      self.variables['moving_variance'] = moving_variance

  def add_variable_by_dict(self, var_dict):
    for k, v in var_dict.items():
      # assert key
      self.variables[k] = v

class Conv2DStructure(object):
  """Structure that records all the variables (kernel and bias) in a single convolution layer."""
  def __init__(self, name, op_type):
    self.name = name
    self.op_type = op_type
    self.variables = {}

  def add_variable(self, kernel, bias=None):
    self.variables['kernel'] = kernel
    if bias is not None:
      self.variables['bias'] = bias
    return

class KeyGraph(object):
  """Graph structure that constains only the high-level operations and their topology"""
  def __init__(self, graph):
    self.graph = graph

    self.nodes = {} # keyed by node name, valued by `KeyNode'
    self.nodes_name = [] # record the keys (mainly for order infomation)
    self._extract_key_ops()
    self._extract_unfusedbn_ops()
    self._extract_fusedbn_ops()
    self._extract_tensordot_ops()
    self._complete_connections()
    self._set_nodes()

  def _set_nodes(self):
      for key in self.nodes:
          self.nodes[key].set_input_nodes()
          self.nodes[key].set_output_nodes()


  def add_node(self, node):
    self.nodes[node.name] = node
    self.nodes_name.append(node.name)

  def get_node(self, node_name):
    return self.nodes[node_name]

  def gap_l1_loss(self, summ=False):
    collection = 'gap_l1_loss'
    self._add_gammas_to_collection(collection)
    gamma_ts = list(set(ops.get_collection(collection)))
    if summ:
        import tensorflow as tf
        for var in gamma_ts:
            tf.summary.histogram('hist_%s' % var.op.name, var)
    l1_loss = math_ops.add_n([\
        math_ops.reduce_sum(math_ops.abs(var))\
        for var in gamma_ts])
    return l1_loss

  def _add_gammas_to_collection(self, collection):
    for node in self.nodes.values():
      if node.op_type == 'BN':
        gamma = node.variables['gamma']
        g_consumers = gamma.outputs[0].consumers()
        for c in g_consumers:
          if (c.node_def.op == 'Identity') or (c.node_def.op == 'ReadVariableOp'):
            gamma_read = c.outputs[0]
            break
        ops.add_to_collection(collection, gamma_read)

  def gap_init(self):
    """ Environment preparation for GAP. """
    self.gamma_structures = {}
    self.bn_vars = {} # keyed by gamma name
    self.conv2d_vars = {} # keyed by conv2d name
    self.biasadd_vars = {}
    self.split_nodes = []
    self._extract_gammas()
    self._group_corr_gammas()

    self._extract_bn_variables()
    self._extract_conv2d_variables()
    self._extract_biasadd_variables()
    self._extract_split_nodes()

  def gap(self, model_weight_path, gap_var_path, mode='gap', slim_ratio=0.5,
          mask_len=False, full_save=False, full_save_path=None, var_scope='', ver=1):
    """Function for the pruning step for GAP (channel level) or Channel Pruning"""
    data = self._load_weights(model_weight_path, full_save, var_scope)
    # save the unpruned model weights
    if full_save:
        with gfile.Open(full_save_path, 'wb') as f:
            pickle.dump(data, f)
        exit(-1)

    self._concise_graph_by_var_list(data.keys())

    self.gap_init()
    masks = self.gap_get_mask(data, mode, slim_ratio, mask_len, ver)
    pruned_data = self.gap_get_pruned_variables(data, masks)

    if gap_var_path:
      with gfile.Open(gap_var_path, 'wb') as f:
        pickle.dump(pruned_data, f)

    return pruned_data

  def gap_get_mask(self, data, mode='gap', slim_ratio=0.5, mask_len=False, ver=1):
    if mode == 'gap':
      initial_masks = self.gap_get_initial_mask(data, slim_ratio=slim_ratio, mask_len=mask_len)
    elif mode == 'chp':
      initial_masks = self.chp_get_initial_masks(data)

    masks = self.gap_propagate_mask(data, initial_masks)
    """
    # stage-1
    masks = initial_masks
    group_bn = ['resnet_model/batch_normalization_1/gamma',
                'resnet_model/batch_normalization_4/gamma',
                'resnet_model/batch_normalization_7/gamma',
                'resnet_model/batch_normalization_10/gamma']
    mask_t = np.logical_or(masks[group_bn[0]], masks[group_bn[1]])
    mask_t = np.logical_or(mask_t, masks[group_bn[2]])
    mask_t = np.logical_or(mask_t, masks[group_bn[3]])
    for gname in group_bn:
         masks[gname] = mask_t

    # stage-2
    group_bn = ['resnet_model/batch_normalization_11/gamma',
                'resnet_model/batch_normalization_14/gamma',
                'resnet_model/batch_normalization_17/gamma',
                'resnet_model/batch_normalization_20/gamma',
                'resnet_model/batch_normalization_23/gamma']
    mask_t = np.logical_or(masks[group_bn[0]], masks[group_bn[1]])
    mask_t = np.logical_or(mask_t, masks[group_bn[2]])
    mask_t = np.logical_or(mask_t, masks[group_bn[3]])
    mask_t = np.logical_or(mask_t, masks[group_bn[4]])
    for gname in group_bn:
         masks[gname] = mask_t

    # stage-3
    group_bn = ['resnet_model/batch_normalization_24/gamma',
                'resnet_model/batch_normalization_27/gamma',
                'resnet_model/batch_normalization_30/gamma',
                'resnet_model/batch_normalization_33/gamma',
                'resnet_model/batch_normalization_36/gamma',
                'resnet_model/batch_normalization_39/gamma',
                'resnet_model/batch_normalization_42/gamma']
    mask_t = np.logical_or(masks[group_bn[0]], masks[group_bn[1]])
    mask_t = np.logical_or(mask_t, masks[group_bn[2]])
    mask_t = np.logical_or(mask_t, masks[group_bn[3]])
    mask_t = np.logical_or(mask_t, masks[group_bn[4]])
    mask_t = np.logical_or(mask_t, masks[group_bn[5]])
    mask_t = np.logical_or(mask_t, masks[group_bn[6]])
    for gname in group_bn:
         masks[gname] = mask_t

    # stage-4
    group_bn = ['resnet_model/batch_normalization_43/gamma',
                'resnet_model/batch_normalization_46/gamma',
                'resnet_model/batch_normalization_49/gamma',
                'resnet_model/batch_normalization_52/gamma']
    mask_t = np.logical_or(masks[group_bn[0]], masks[group_bn[1]])
    mask_t = np.logical_or(mask_t, masks[group_bn[2]])
    mask_t = np.logical_or(mask_t, masks[group_bn[3]])
    for gname in group_bn:
         masks[gname] = mask_t
    """
    # post-make sure mask_len % _BASE_LEN == 0
    if _BASE_LEN > 0 and mask_len:
        for gamma_name, _ in self.bn_vars.items():
            mask = masks[gamma_name]
            gamma = data[gamma_name]
            inx = np.argsort(-gamma) # descending
            # processing of channel mask
            find_num = re.findall(r"\d+", gamma_name)
            if not find_num:
                bn_layer_num = 0
            else:
                bn_layer_num = int(find_num[0])

            # the first res group is dont touch
            if ver == 1: # 50-v1
                if bn_layer_num <= 10:
                    mask[:] = True
            elif ver == 261: # 26-v1
                if bn_layer_num <= 14:
                    mask[:] = True
            elif ver == 14: # 50-v1d
                if bn_layer_num <= 12:
                    mask[:] = True
            elif ver == 2614: # 26-v1d
                if bn_layer_num <= 9:
                    mask[:] = True

            # match with 32-fold
            mask_len = np.sum(mask)
            mask_mod = mask_len % _BASE_LEN
            if mask_mod > 0:
                j = 0
                for i in range(len(mask)):
                    if not mask[inx[i]]:
                        mask[inx[i]] = True
                        j += 1
                    if j == (_BASE_LEN-mask_mod):
                        break

            if (ver == 261):
                mask_len_t = 0
                mask_len_c = 0
                inx_inv = np.argsort(gamma) # ascending
                if (int(slim_ratio*100)==0):
                    if bn_layer_num in [26, 27]:
                        mask_len_t = 128
                    elif bn_layer_num in [23, 24]:
                        mask_len_t = 256
                    elif bn_layer_num in [15, 18, 21, 22, 25, 28]:
                        mask_len_t = 384

                elif (int(slim_ratio*100)==50):
                    if bn_layer_num in [17, 19]:
                        mask_len_t = 0
                        mask[:] = True
                    elif bn_layer_num in [23]:
                        mask_len_t = 128
                    elif bn_layer_num in [22, 25, 28]:
                        mask_len_c = 64

                if mask_len_t > 0:
                    j = 0
                    for i in range(len(inx_inv)):
                        if mask[inx_inv[i]]:
                            mask[inx_inv[i]] = False
                            j += 1
                        if j == mask_len_t:
                            break

                if mask_len_c > 0:
                    j = 0
                    for i in range(len(mask)):
                        if not mask[inx[i]]:
                            mask[inx[i]] = True
                            j += 1
                        if j == mask_len_c:
                            break

            masks[gamma_name] = mask
    for gamma_name, _ in self.bn_vars.items():
        mask = masks[gamma_name]
        print('post-make sure mask len:', gamma_name, np.sum(mask))
    return masks

  def _load_weights(self, model_weight_path, full_save=False, var_scope=''):
    """Function to load weights from give model data path `model_weight_path'. """
    reader = pywrap_tensorflow.NewCheckpointReader(model_weight_path)
    var_to_shape_map = reader.get_variable_to_shape_map()
    data = dict()
    for name in var_to_shape_map:
      if ('teach' in name) or ('tower' in name):
          continue
      tensor = reader.get_tensor(name)
      print(name)
      if var_scope and full_save:
        name = name.replace(name.split('/')[0], var_scope)
      data[name] = tensor

    return data

  def _concise_graph_by_var_list(self, var_list):
    black_list = []
    white_list = []

    # get the initial white_list/black_list
    for name in self.nodes_name:
      node = self.nodes[name]
      tag = False
      for var in node.variables.values():
        if type(var) is not int and var.node_def.name not in var_list:
          tag = True
      if tag:
        black_list.append(name)
      else:
        white_list.append(name)

    while(True):
      num_white = len(white_list)
      for name in white_list[:]:
        node = self.nodes[name]
        input_tag = [input in black_list for input in node.input_nodes]
        if len(node.input_nodes) != 0 and sum(input_tag) == len(node.input_nodes):
          black_list.append(name)
          white_list.remove(name)
          continue
        output_tag = [output in black_list for output in node.output_nodes]
        if len(node.output_nodes) != 0 and sum(output_tag) == len(node.output_nodes):
          black_list.append(name)
          white_list.remove(name)

      # white_list no longer change
      if len(white_list) == num_white:
        break


    # clean the key_graph
    for name in self.nodes_name[:]:
      if name in black_list:
        self.nodes_name.remove(name)
        self.nodes.pop(name)
      else:
        for input, gamma in zip(node.input_nodes, node.input_gammas):
          if input in black_list:
            node.input_nodes.remove(input)
            node.input_gammas.remove(gamma)
        for output, gamma in zip(node.output_nodes, node.output_gammas):
          if output in white_list:
            node.output_nodes.remove(output)
            node.output_gammas.remove(gamma)
    return


  def gap_get_threshold(self, data, slim_ratio=0.5):
    gammas = []
    for gamma_name in self.bn_vars.keys():
      gamma = data[gamma_name]
      gammas = np.hstack((gammas, gamma))
    gammas.sort()
    num_zeros = np.float(np.sum(gammas < 1e-5))
    sparsity = num_zeros/gammas.shape[0]
    print("Sparsity of BN is: {:.4f}".format(sparsity))
    thres = gammas[int(slim_ratio*len(gammas))]
    return thres

  def chp_get_initial_masks(self, data):
    masks = dict()
    for gamma_name in self.bn_vars.keys():
      gamma = data[gamma_name]
      masks[gamma_name] = np.zeros(gamma.shape, dtype = bool)

    for key in self.conv2d_vars.keys():
      kernel_name = self.conv2d_vars[key].variables['kernel']
      if kernel_name not in data:
        continue
      kernel_value = data[kernel_name]
      nnzs_mask = np.sum(np.abs(kernel_value), axis=(0,1,3)) > 0
      input_gammas = self.nodes[key].input_gammas
      for input_gamma in input_gammas:
        input_mask = input_gamma.get_value(masks)
        input_mask = np.logical_or(input_mask, nnzs_mask)
        input_gamma.update_value(masks, input_mask)
    return masks

  def gap_get_initial_mask(self, data, slim_ratio, mask_len, narrow=False):
    """ Function to get the original mask given pruning ratio. """
    thres = self.gap_get_threshold(data, slim_ratio=slim_ratio)

    masks = dict()
    # add the value of gammas with `Plain' mode into `masks'
    for gamma_name in self.bn_vars.keys():
      gamma = data[gamma_name]
      mask = gamma >= thres

      if np.sum(mask) == 0:
        inx = np.argsort(-gamma)
        min_len = int(max(0.1*len(gamma), 1))
        mask[inx[:min_len]] = True

      print('before-make sure mask len:', gamma_name, np.sum(mask))

      if _BASE_LEN > 0 and mask_len:
          # pre-make sure mask_len % _BASE_LEN == 0
          mask_len = np.sum(mask)
          mask_mod = mask_len % _BASE_LEN
          if mask_mod > 0:
              #if mask_len < 96:
              mask_len += (_BASE_LEN - mask_mod)
              #mask_len = 64 if mask_len < 64 else mask_len
              #else:
              #  mask_len -= mask_mod
              inx = np.argsort(-gamma)
              mask[inx[:mask_len]] = True
      masks[gamma_name] = mask
      print('pre-make sure mask len:', gamma_name, np.sum(mask))
    return masks

  def gap_propagate_mask(self, data, masks):
    """Function to propagate masks"""
    # add the value of gammas with `AllSelect' mode into `masks'
    for name, gamma in self.gamma_structures.items():
      if gamma.gamma_mode == 'AllSelect':
        value = gamma.get_value(masks)
        masks[name] = value

    for name, gamma in self.gamma_structures.items():
      if gamma.gamma_mode == 'Split':
        mask = gamma.get_value(masks)
        if np.sum(mask) == 0:
          mask = np.array(mask)
          value = gamma.get_value(data)
          inx = np.argsort(-np.array(value))
          min_len = int(max(0.1*len(value), 1))
          mask[inx[:min_len]] = True
          gamma.update_value(masks, list(mask))

    for name in self.roots:
      self._propagate_root_gamma(self.nodes[name], masks)

    for name in self.ending:
      self._propagate_ending_gamma(self.nodes[name], masks)

    # update the mask based on correlation: `logical_or'
    in_group_indices = []
    for group_inx in range(len(self.gamma_group_inx2names)):
      if len(self.gamma_group_inx2names[group_inx]) <= 1:
        continue
      in_group_indices.append(group_inx)
      updated_mask = self._get_group_mask(masks, group_inx)
      self._update_group_mask(masks, group_inx, updated_mask)

    # update the mask based on duplicate BNs: `logical_and'
    for inx in range(len(self.gamma_duplicate_inx2names)):
      duplicate_group = self.gamma_duplicate_inx2names[inx]
      updated_mask = True
      for name in duplicate_group:
        cur_mask = self.gamma_structures[name].get_value(masks)
        updated_mask = np.logical_and(cur_mask, updated_mask)

      for name in duplicate_group:
        group_inx = self.gamma_group_name2inx[name]
        if group_inx != -1 and len(self.gamma_group_inx2names[group_inx]) > 1:
          self._update_group_mask(masks, group_inx, updated_mask)
        else:
          self.gamma_structures[name].update_value(masks, updated_mask)

    return masks

  def _propagate_ending_gamma(self, node, masks):
    """ Propagate the ending gamma `Allselect' until encountering `Conv2D' or `MatMul'. """
    if node.op_type == 'Conv2D' or node.op_type == 'MatMul':
      return
    for i in range(len(node.input_nodes)):
      pre_node = self.nodes[node.input_nodes[i]]
      if pre_node.op_type == 'BN':
        gamma_name = pre_node.output_gammas[0].name
        masks[gamma_name] = np.logical_or(masks[gamma_name], True)

      else:
        self._propagate_ending_gamma(pre_node, masks)

  def _propagate_root_gamma(self, node, masks):
    """ Propagate the root gamma `Allselect' until encountering `Conv2D' or `MatMul'. """
    if node.op_type == 'Conv2D' or node.op_type == 'MatMul':
      return
    for i in range(len(node.output_nodes)):
      post_node = self.nodes[node.output_nodes[i]]
      if post_node.op_type == 'BN':
        gamma_name = post_node.output_gammas[0].name
        masks[gamma_name] = np.logical_or(masks[gamma_name], True)
      else:
        self._propagate_root_gamma(post_node, masks)

  def _get_group_mask(self, data_dict, group_inx):
    """ Get the logical_or mask concerns withing a correlated group. """
    updated_mask = False
    for name in self.gamma_group_inx2names[group_inx]:
      gamma = self.gamma_structures[name]
      cur_mask = False
      if gamma.gamma_mode == 'Split':
        corr_gamma_inx = self.gamma_group_name2inx[gamma.corr_gammas[0].name]
        if corr_gamma_inx == -1 or \
          len(self.gamma_group_inx2names[corr_gamma_inx]) == 1:
          cur_mask = gamma.get_value(data_dict)
        else:
          corr_mask = self._get_group_mask(data_dict, corr_gamma_inx)
          cur_mask = corr_mask[gamma.start_inx:gamma.end_inx]
      elif gamma.gamma_mode == 'Concat':
        cur_mask = []
        for corr_gamma in gamma.corr_gammas:
          corr_gamma_inx = self.gamma_group_name2inx[corr_gamma.name]
          if corr_gamma_inx == -1 or \
              len(self.gamma_group_inx2names[corr_gamma_inx]) == 1:
            corr_mask = corr_gamma.get_value(data_dict)
          else:
            corr_mask = self._get_group_mask(data_dict, corr_gamma_inx)
          cur_mask = cur_mask.extend(corr_mask)
      else:
        cur_mask = gamma.get_value(data_dict)
      updated_mask = np.logical_or(updated_mask, cur_mask)

    return list(updated_mask)


  def _update_group_mask(self, data_dict, group_inx, updated_mask):
    """ Update the logical_or mask concerns withing a correlated group. """
    for name in self.gamma_group_inx2names[group_inx]:
      gamma = self.gamma_structures[name]
      if gamma.gamma_mode == 'Split':
        cur_mask = gamma.get_value(data_dict)
        cur_mask = np.logical_or(updated_mask, cur_mask)
        gamma.update_value(data_dict, cur_mask)
        corr_gamma_inx = self.gamma_group_name2inx[gamma.corr_gammas[0].name]
        if corr_gamma_inx != -1 \
            and len(self.gamma_group_inx2names[corr_gamma_inx]) > 1:
          corr_gamma = self.gamma_structures[gamma.corr_gammas[0].name]
          corr_mask = corr_gamma.get_value(data_dict)
          self._update_group_mask(data_dict, corr_gamma_inx, corr_mask)
      elif gamma.gamma_mode == 'Concat':
        cur_mask = gamma.get_value(data_dict)
        cur_mask = np.logical_or(updated_mask, cur_mask)
        gamma.update_value(data_dict, cur_mask)
        cur_mask = []
        for corr_gamma in gamma.corr_gammas:
          corr_gamma_inx = self.gamma_group_name2inx[corr_gamma.name]
          if corr_gamma_inx != -1 \
              and len(self.gamma_group_inx2names[corr_gamma_inx]) > 1:
            corr_mask = corr_gamma.get_value(data_dict)
            self._update_group_mask(data_dict, corr_gamma_inx, corr_mask)
      else:
        cur_mask = gamma.get_value(data_dict)
        cur_mask = np.logical_or(updated_mask, cur_mask)
        gamma.update_value(data_dict, cur_mask)


  def gap_get_pruned_variables(self, data, masks):
    """ Function to get the pruned variables given the original data dict `data' and pruning mask.
    Args:
      data: orginal data dict structured by
          {key: variable name, value: weight value}
      masks: pruning mask structured by
          {key: gamma name, value: mask for the corresponding gamma variable}
    Return:
      pruned_data: pruned data dict contains
        1) standart variable: {key: variable name, value: pruned weight value}
        2) split shape structured by
            { key: split scope name+`/split',
              value: split_subdict}
            and the `split_subdict' is structured by:
                { key: index,
                  value: dict{'Tag': False, 'Value': split output shape}}
           --> split_subdict with keys `index' mainly to
               process multiple `Split' opertaion within the same scope
    """

    pruned_data = dict()

    # prune the BN paras
    for gamma_name, bn_structure in self.bn_vars.items():
      mask = masks[gamma_name]

      gamma = bn_structure.variables['gamma']
      value = data[gamma]
      pruned_data[gamma] = value[mask]

      #print(gamma_name + ': from ' + str(len(mask)) + ' to ' + str(np.sum(mask)))

      beta = bn_structure.variables['beta']
      value = data[beta]
      pruned_data[beta] = value[mask]

      moving_mean = bn_structure.variables['moving_mean']
      value = data[moving_mean]
      pruned_data[moving_mean] = value[mask]

      moving_variance = bn_structure.variables['moving_variance']
      value = data[moving_variance]
      pruned_data[moving_variance] = value[mask]

    # get split paras
    for name in self.split_nodes:
      output_gammas = self.nodes[name].output_gammas
      output_shape = []
      for gamma in output_gammas:
        mask = gamma.get_value(masks)
        output_shape.append(np.sum(mask))

      inx = name.rfind('split')
      if len(name[inx:].split('_')) > 1:
        split_name = name.rsplit('_', 1)[0]
        split_inx = int(name.rsplit('_', 1)[1])
        if not (split_name in pruned_data):
            pruned_data[split_name] = dict()
        pruned_data[split_name][split_inx] = {
            'Tag': False,
            'Value': output_shape,
            }
      else:
        split_name = name
        pruned_data[split_name] = dict()
        pruned_data[split_name][0] = {
            'Tag': False,
            'Value': output_shape,
            }

    for name, conv2d in self.conv2d_vars.items():
      input_gammas = self.nodes[name].input_gammas
      output_gammas = self.nodes[name].output_gammas

      if input_gammas:
        input_mask = input_gammas[0].get_value(masks)
      else:
        input_mask = True
      if output_gammas:
        output_mask = output_gammas[0].get_value(masks)
      else:
        output_mask = True


      kernel = data[conv2d.variables['kernel']]

      original_shape = kernel.shape
      if isinstance(input_mask, list):
        if len(kernel.shape) == 4:
          kernel = kernel[:, :, input_mask, :]
        elif len(kernel.shape) == 2:
          kernel = kernel[input_mask, :]
      if conv2d.op_type in  ['Conv2D', 'MatMul'] and isinstance(output_mask, list):
        if len(kernel.shape) == 4:
          kernel = kernel[:, :, :, output_mask]
        elif len(kernel.shape) == 2:
          kernel = kernel[:, output_mask]
      pruned_data[conv2d.variables['kernel']] = kernel
      #print(name + ': from ' + str(original_shape) + ' to ' + str(kernel.shape))

      """
      if conv2d.variables.has_key('bias'):
        bias = data[conv2d.variables['bias']]
        bias = bias[output_mask]
        pruned_data[conv2d.variables['bias']] = bias
      """
    for name, bias in self.biasadd_vars.items():
      gammas = self.nodes[name].input_gammas
      if gammas:
        mask = gammas[0].get_value(masks)
      else:
        mask = True

      original_shape = data[bias].shape
      pruned_data[bias] = data[bias][mask]
      #print(name + ': from ' + str(original_shape) + ' to ' + str(pruned_data[bias].shape))


    data_keys = set(data.keys())
    pruned_data_keys = set(pruned_data.keys())
    untouched_keys = data_keys - pruned_data_keys
    for key in list(untouched_keys):
      if not IsOptimizerVars(key):
        pruned_data[key] = data[key]
    return pruned_data


  def _add_key_op(self, node):
    """ Construct a key high-level operation `KeyNode'. """
    key_node = KeyNode(node.node_def.name, node.node_def.op)
    if node.node_def.op == 'Conv2D':
      # add variable
      variables = self._find_root_variable(node.inputs[1].op)
      kernel = variables[0]
      key_node.add_variable({'kernel': kernel})
      # add input_size, output_size
      kernel_shape = node.inputs[1].get_shape().as_list()
      input_size = kernel_shape[-2]
      output_size = kernel_shape[-1]
    elif node.node_def.op == 'DepthwiseConv2dNative':
      # add variable
      variables = self._find_root_variable(node.inputs[1].op)
      kernel = variables[0]
      key_node.add_variable({'kernel': kernel})
      # add input_size, output_size
      input_size = node.inputs[0].get_shape().as_list()[-1]
      output_size = node.outputs[0].get_shape().as_list()[-1]
    elif node.node_def.op == 'BiasAdd':
      # add variable
      variables = self._find_root_variable(node.inputs[1].op)
      bias = variables[0]
      key_node.add_variable({'bias': bias})
      # add input_size, output_size
      input_size = node.inputs[0].get_shape().as_list()[-1]
      output_size = node.outputs[0].get_shape().as_list()[-1]
    elif node.node_def.op == 'Split':
      # add variable
      axis = node.inputs[0].op.node_def.attr["value"].tensor.int_val[0]
      key_node.add_variable({'axis': axis})
      # add input_size, output_size
      num_split = node.node_def.attr['num_split'].i
      input_size = node.inputs[1].get_shape().as_list()[-1]
      if axis in _CHANNEL_AXIS:
        output_size = np.ones(num_split, dtype=np.int32)*(input_size//num_split)
      else:
        output_size = np.ones(num_split, dtype=np.int32)*(input_size)
      output_size = list(output_size)
    elif node.node_def.op == 'SplitV':
      # add variable
      axis = node.inputs[2].op.node_def.attr["value"].tensor.int_val[0]
      key_node.add_variable({'axis': axis})
      # add input_size, output_size
      num_split = node.node_def.attr['num_split'].i
      input_size = node.inputs[0].get_shape().as_list()[-1]
      size_split = np.fromstring( \
          node.inputs[1].op.node_def.attr["value"].tensor.tensor_content,
          dtype=np.int32)
      if axis in _CHANNEL_AXIS:
        output_size = size_split
      else:
        output_size = np.ones(num_split, dtype=np.int32)*(input_size)
      output_size = list(output_size)
    elif node.node_def.op == 'ConcatV2':
      # add variable
      axis_node = node.inputs[-1].op.node_def
      axis = axis_node.attr['value'].tensor.int_val[0]
      key_node.add_variable({'axis': axis})
      # add input_size, output_size
      output_size = node.outputs[0].get_shape().as_list()[-1]
      input_size = []
      for i in range(len(node.inputs)-1):
        input_size.append(node.inputs[i].get_shape().as_list()[-1])
    elif node.node_def.op == 'Add' or node.node_def.op in _KEY_OPS_NONCHANNEL:
      # add input_size, output_size
      output_size = node.outputs[0].get_shape().as_list()[-1]
      input_size = output_size
    elif node.node_def.op == 'MatMul':
      variables = self._find_root_variable(node.inputs[1].op)
      kernel = variables[0]
      key_node.add_variable({'kernel': kernel})
      # add input_size, output_size
      input_size = node.inputs[0].get_shape().as_list()[-1]
      output_size = node.outputs[0].get_shape().as_list()[-1]
    else:
      input_size = None
      output_size = None
      #assert 'Unrecognized op type: %s'%node.node_def.op

    if input_size:
      key_node.set_input_size(input_size)
    if output_size:
      key_node.set_output_size(output_size)

    # inputs
    for i in node.inputs:
      i = i.op
      if i.node_def.op in _KEY_OPS or i.node_def.name in self.nodes:
        key_node.add_input_nodes(i.node_def.name)

    def find_input_nodes(key_node, node):
        for i in node.inputs:
          i = i.op
          if i.node_def.op in _KEY_OPS or i.node_def.name in self.nodes:
            key_node.add_input_nodes(i.node_def.name)
          elif i.node_def.op == 'Identity' or i.node_def.op == 'Pad' or i.node_def.op == 'AvgPool' \
               or i.node_def.op == 'Mean' or i.node_def.op == 'Reshape':
            find_input_nodes(key_node, i)

    if not key_node.input_nodes:
        find_input_nodes(key_node, node)

    # outputs
    for t in node.outputs:
      for o in t.consumers():
        if o.node_def.op in _KEY_OPS or o.node_def.name in self.nodes:
          key_node.add_output_nodes(o.node_def.name)

    def find_output_nodes(key_node, node):
        for t in node.outputs:
          for o in t.consumers():
              if o.node_def.op in _KEY_OPS or o.node_def.name in self.nodes:
                key_node.add_output_nodes(o.node_def.name)
              elif o.node_def.op == 'Identity' or o.node_def.op == 'Pad' or i.node_def.op == 'AvgPool' \
                   or o.node_def.op == 'Mean' or o.node_def.op == 'Reshape':
                find_output_nodes(key_node, o)

    if not key_node.output_nodes:
        find_output_nodes(key_node, node)

    self.add_node(key_node)
    return

  def _find_fusedbn_variables(self, node, is_training=False):
    """ Find the variables correlated the BN layer with the BN is realized by `FusedBatchNorm' """
    inputs = []
    for i in node.inputs:
      if i.op.node_def.op == 'Switch':
        inputs.append(i.op.inputs[0].op)
      else:
        inputs.append(i.op)

    # gamma
    variables = []
    self._find_root_variable(inputs[1], variables)
    gamma = variables[0]
    # beta
    variables = []
    self._find_root_variable(inputs[2], variables)
    beta = variables[0]
    if not is_training: # only inference phase
      # moving_mean
      variables = []
      self._find_root_variable(inputs[3], variables)
      moving_mean = variables[0]
      # moving_variance
      variables = []
      self._find_root_variable(inputs[4], variables)
      moving_variance = variables[0]
    else: # only training phase
      mean_output = node.outputs[1]
      variance_output = node.outputs[2]
      # moving_mean
      for c in mean_output.consumers():
        if c.node_def.op == 'Sub':
          source = c.inputs[0].op
          variables = []
          self._find_root_variable(source, variables)
          moving_mean = variables[0]
          break
      # moving_variance
      for c in variance_output.consumers():
        if c.node_def.op == 'Sub':
          source = c.inputs[0].op
          variables = []
          self._find_root_variable(source, variables)
          moving_variance = variables[0]
          break

    bn_variables = {
        'gamma': gamma,
        'beta': beta,
        'moving_mean': moving_mean,
        'moving_variance': moving_variance
    }
    return bn_variables

  def _extract_key_ops(self):
    # extract key ops except `BN' and `Add'
    for node in self.graph.get_operations():
      if IsShortcutOp(node) or node.type in _KEY_OPS and _GRADIENTS_SCOPE not in node.name:
        self._add_key_op(node)
    return

  def _find_root_variable(self, node, variables=None):
    """ Find the variables that have access to the provided node. """
    if variables is None:
      variables = []
    if node.node_def.op == "VariableV2" or node.node_def.op == "Variable" or node.node_def.op == 'VarHandleOp':
      variables.append(node)
      return variables
    if node.node_def.op == 'Conv2D':
      return variables

    for n in node.inputs:
      variables = self._find_root_variable(n.op, variables)
    return variables


  def _extract_fusedbn_ops(self):
    """ Extract the BN layers implemented by `FusedBatchNomr'. """
    explored_bns = []
    for node in self.graph.get_operations():
      if node.node_def.op != 'FusedBatchNorm':
        continue
      if node.node_def.name in explored_bns:
        continue


      explored_bns.append(node.node_def.name)

      # todo: process scale=False, no `gamma'
      # todo: process bias=False, no `bias'
      train_inference = False
      featuremap = node.outputs[0]
      #consumers = [c for c in featuremap.consumers() if c.type not in _SKIPPED_OPS]
      merge_op = None
      for c in featuremap.consumers():
        if c.node_def.op == 'Merge':
          train_inference = True
          merge_op = c

      def get_post_op(op):
        post_op = []
        for c in op.outputs[0].consumers():
          if c.type in _SKIPPED_OPS:
            continue
          elif c.type ==  'Identity' and c.control_inputs or c.type in _SHORTCUT_OPS:
            post_op.extend(c)
          else:
            post_op.append(c)
        return post_op

      if train_inference:
        inx = 0
        for i in merge_op.inputs:
          if i.op.node_def.name == node.node_def.name:
            break
          else:
            inx += 1
        # inputs[3] is moving_mean
        if node.inputs[3].op.node_def.op == 'Const':
          # only two inputs, inx is either 0 or 1
          inference_bn = merge_op.inputs[1-inx].op
        else:
          inference_bn = node

        explored_bns.append(merge_op.inputs[1-inx].op.node_def.name)
        # get variables
        variables = self._find_fusedbn_variables(\
                          inference_bn,
                          is_training=False)

        # get post_op and pre_op, removing the `Switch' and `Merge' ops
        post_op = get_post_op(merge_op)
        #post_op = merge_op.outputs[0].consumers()[0]
        pre_op = inference_bn.inputs[0].op.inputs[0].op

        """
        # process of post_op, control_dependency will introduce Identity, which may be shown when force update moving_average
        if post_op.node_def.op == 'Identity' \
            and post_op.control_inputs:
          post_op = post_op.outputs[0].consumers()[0]
        """

      else:
        pre_op = node.inputs[0].op
        post_op = get_post_op(node)
        #post_op = node.outputs[0].consumers()[0]
        #if post_op.node_def.op == 'Identity' \
        #    and post_op.control_inputs:
        #  post_op = post_op.outputs[0].consumers()[0]
        # only train phase
        variables = self._find_root_variable(node.inputs[3].op)
        #if node.inputs[2].op.node_def != 'Const':
        if len(variables) == 0:
          variables = self._find_fusedbn_variables(node, is_training=True)
        # only inference phase
        else:
          variables = self._find_fusedbn_variables(node, is_training=False)

      while pre_op.node_def.op in _SHORTCUT_OPS:
        pre_op = pre_op.inputs[0].op

      #while post_op.node_def.op in _SHORTCUT_OPS:
      #  post_op = post_op.outputs[0].consumers()[0]

      pre_op_name = pre_op.name
      post_op_names = [op.name for op in post_op]
      output_size = int(pre_op.outputs[0].get_shape().as_list()[-1])
      input_size = output_size
      key_node = KeyNode(node.node_def.name,
                         'BN',
                         input_size,
                         output_size,
                         pre_op_name,
                         post_op_names,
                         variables)

      self._add_bn_to_graph(key_node, pre_op, post_op)


  def _extract_tensordot_ops(self):
    pattern = TensorDotPattern()
    pattern_map = pattern.pattern_map
    first_key = pattern.first_key
    numinputs = pattern.numinputs
    numoutputs = pattern.numoutputs
    entry_op_check = pattern.entry_op_check

    pm = PatternMatch(pattern_map, entry_op_check, first_key, numinputs, numoutputs)
    matched_tensordot = pm.match(self.graph)

    for nodes in matched_tensordot:
      name = nodes['matmul'][0].name
      key_node = self.nodes[name]
      pre_op = nodes['transpose'][0].inputs[0].op
      while pre_op.node_def.op in _SHORTCUT_OPS:
        pre_op = pre_op.inputs[0].op
      post_op = [op for op in nodes['reshape_2'][0].outputs[0].consumers() if op not in _SKIPPED_OPS]
      key_node.add_input_nodes(pre_op.name)
      key_node.add_output_nodes([op.name for op in post_op])
      self._add_bn_to_graph(key_node, pre_op, post_op)


  def _extract_unfusedbn_ops(self):
    """ Extract unfused BN ops by pattern matching """
    pattern = BatchNormPattern()
    pattern_map = pattern.pattern_map
    first_key = pattern.first_key
    numinputs = pattern.numinputs
    numoutputs = pattern.numoutputs
    entry_op_check = pattern.entry_op_check
    pm = PatternMatch(pattern_map, entry_op_check, first_key, numinputs, numoutputs)
    # matched_bns: the matched BN, while each BN is represented by a set of nodes
    matched_bns = pm.match(self.graph)

    for nodes in matched_bns:
      name = self._get_unfusedbn_name(nodes)
      variables = self._find_unfusedbn_variables(nodes)
      pre_op = self._find_unfusedbn_pre_op(nodes)
      post_op = self._find_unfusedbn_post_op(nodes)
      output_size = int(pre_op.outputs[0].get_shape().as_list()[-1])
      input_size = output_size
      key_node = KeyNode(name,
                         'BN',
                         input_size,
                         output_size,
                         pre_op.node_def.name,
                         post_op.node_def.name,
                         variables)
      self._add_bn_to_graph(key_node, pre_op, post_op)

  def _add_bn_to_graph(self, key_node, pre_op, post_op):
    """ Add BN node and its connection to the key graph topology. """
    name = key_node.name
    self.add_node(key_node)

    # processing connection concerning pre_op
    if pre_op.node_def.name in self.nodes_name:
      pre_node = self.nodes[pre_op.node_def.name]
      pre_node.add_output_nodes(name)
    elif _GRADIENTS_SCOPE not in pre_op.name and pre_op.type not in _SKIPPED_OPS:
      self._add_key_op(pre_op)
      pre_node = self.nodes[pre_op.node_def.name]
      pre_node.add_output_nodes(name)
      # update outputs of pre nodes
      for i in pre_node.input_nodes:
        pre_pre_node = self.nodes[i]
        pre_pre_node.add_output_nodes(pre_node.name)

    # processing connection concerning post_op
    for op in post_op:
      if op.node_def.name in self.nodes_name:
        post_node = self.nodes[op.node_def.name]
        post_node.add_input_nodes(name)
      elif _GRADIENTS_SCOPE not in op.name and op.type not in _SKIPPED_OPS:
        self._add_key_op(op)
        post_node = self.nodes[op.node_def.name]
        post_node.add_input_nodes(name)
        # update outputs of pre_nodes
        for i in post_node.input_nodes:
          if i != name:
            pre_post_node = self.nodes[i]
            pre_post_node.add_output_nodes(post_node.name)
        # update inputs of post nodes
        for i in post_node.output_nodes:
          post_post_node = self.nodes[i]
          post_post_node.add_input_nodes(post_node.name)


  def _complete_connections(self):
    for name, node in self.nodes.items():
      # input
      for input in node.input_nodes[:]:
        if input in self.nodes:
          pre_node = self.nodes[input]
          if name not in pre_node.output_nodes:
            pre_node.output_nodes.append(name)
        else:
          node.input_nodes.remove(input)
      # output
      for output in node.output_nodes[:]:
        if output in self.nodes:
          post_node = self.nodes[output]
          if name not in post_node.input_nodes:
            post_node.input_nodes.append(name)
        else:
          node.output_nodes.remove(output)


  def _get_unfusedbn_name(self, nodes):
    bn_name = nodes['add_1'][0].node_def.name
    inx = bn_name.rfind('/')
    bn_name = bn_name[:inx]
    return bn_name

  def _find_unfusedbn_post_op(self, nodes):
    add_1 = nodes['add_1'][0]
    return [op for op in add_1.outputs[0].consumers() if op.name in self.nodes]

  def _find_unfusedbn_pre_op(self, nodes):
    mul_1 = nodes['mul_1'][0]
    pre_op = mul_1.inputs[0].op
    while pre_op.node_def.op in _SHORTCUT_OPS:
      pre_op = pre_op.inputs[0].op
    return pre_op

  def _find_unfusedbn_variables(self, nodes):
    """ Find the variables correlated the BN layer with the BN is implemented by unfused format """
    # gamma
    mul = nodes['mul'][0]
    gamma_source = mul.inputs[1].op
    variables = self._find_root_variable(gamma_source)
    gamma = variables[0]

    # beta
    add_1 = nodes['add_1'][0]
    sub = add_1.inputs[1].op
    beta_source = sub.inputs[0].op
    variables = self._find_root_variable(beta_source)
    beta = variables[0]

    # moving_variance
    rsqrt = nodes['rsqrt'][0]
    moving_variance_source = rsqrt.inputs[0].op
    variables = self._find_root_variable(moving_variance_source)
    if not variables:
      add = moving_variance_source
      input_0 = add.inputs[0].op
      if input_0.node_def.op == 'Identity' \
          and input_0.control_inputs:
        moving_assign = input_0.control_inputs[1]
        moving_variance = moving_assign.inputs[0].op
    else:
      moving_variance = variables[0]

    # moving_mean
    mul = nodes['mul'][0]
    mul_consumers = mul.outputs[0].consumers()
    mul_2 = mul_consumers[1]
    moving_mean_source = mul_2.inputs[0].op
    variables = self._find_root_variable(moving_mean_source)
    if not variables:
      input_0 = mul_2.inputs[0].op
      if input_0.node_def.op == 'Identity' \
          and input_0.control_inputs:
        moving_assign = input_0.control_inputs[0]
        moving_mean = moving_assign.inputs[0].op
    else:
      moving_mean = variables[0]

    bn_variables = {
        'gamma': gamma,
        'beta': beta,
        'moving_mean': moving_mean,
        'moving_variance': moving_variance
    }

    return bn_variables

  def _check_gammas_mode(self, gammas):
    if len(gammas) <= 1:
      return True

    mode = gammas[0][0]
    for mode_gamma in gammas:
      if mode_gamma[0] != mode:
        return False
    return True

  def _find_pre_bns_for_add(self, name, pre_bns=None, prefix=None):
    """ Find the connected BN before `Add'. """
    if pre_bns is None:
      pre_bns = []
    if prefix is None:
      prefix = ''

    node = self.nodes[name]
    for i in node.input_nodes:
      if self.nodes[i].op_type == 'BN':
        pre_bns.append([prefix, i])
      elif self.nodes[i].op_type == 'Conv2D' \
          or self.nodes[i].op_type == 'MatMul':
        continue
      else:
        prefix_new = prefix
        if self.nodes[i].op_type != 'Relu' \
            and self.nodes[i].op_type not in prefix:
          prefix_new = prefix + '-' + self.nodes[i].op_type
        self._find_pre_bns_for_add(i, pre_bns, prefix_new)

    return

  def _find_post_bns_for_add(self, name, post_bns=None, prefix=None):
    """ Find the connected BN after `Add'. """
    if post_bns is None:
      post_bns = []
    if prefix is None:
      prefix = ''

    node = self.nodes[name]
    for i in node.output_nodes:
      if self.nodes[i].op_type == 'BN':
        post_bns.append([prefix, i])
      elif self.nodes[i].op_type == 'Conv2D' \
          or self.nodes[i].op_type == 'MatMul':
        continue
      else:
        if self.nodes[i].op_type != 'Relu' \
            and self.nodes[i].op_type not in prefix:
          prefix = prefix + '-' + self.nodes[i].op_type
        self._find_post_bns_for_add(i, post_bns, prefix)
    return

  def _extract_bn_variables(self):
    for key in self.nodes_name:
      node = self.nodes[key]
      if node.op_type == 'BN':
        bn_structure = BNStructure(key)
        bn_structure.add_variable_by_dict(node.variables_name)
        key = node.variables_name['gamma']
        self.bn_vars[key] = bn_structure

  def _extract_conv2d_variables(self):
    """ Extract the variables related to convolution. """
    for key in self.nodes_name:
      node = self.nodes[key]
      if node.op_type in ['Conv2D', 'DepthwiseConv2dNative', 'MatMul']:
        conv2d = Conv2DStructure(key, node.op_type)

        kernel = node.variables['kernel'].node_def.name
        bias = None
        for o in node.output_nodes:
          if self.nodes[o].op_type == 'BiasAdd':
            bias = self.nodes[o].variables['bias'].node_def.name
            break
        conv2d.add_variable(kernel, bias)
        self.conv2d_vars[key] = conv2d


  def _extract_biasadd_variables(self):
    for key in self.nodes_name:
      node = self.nodes[key]
      if node.op_type == 'BiasAdd':
        bias = node.variables['bias'].node_def.name
        self.biasadd_vars[key] = bias



  def _extract_split_nodes(self):
    for key in self.nodes_name:
      node = self.nodes[key]
      if node.op_type in _SPLIT_OPS:
        self.split_nodes.append(key)

  def _extract_add_directions(self):
    """ Function to find BN that affects the `add' node is from input or output or both. """
    self.add_directions = {} # keyed by add node name, value: `in_out', `input', `output'
    for key in self.nodes_name:
      node = self.nodes[key]
      if node.op_type == 'Add':
        pre_bns = []
        post_bns = []
        self._find_pre_bns_for_add(key, pre_bns)
        self._find_post_bns_for_add(key, post_bns)
        if pre_bns and post_bns:
          self.add_directions[key] = ['in_out']
        elif pre_bns:
          self.add_directions[key] = ['input', 'in_out']
        elif post_bns:
          self.add_directions[key] = ['output', 'in_out']

  def _group_corr_gammas(self):
    """ Find the correlated gammas and put them into a group. """
    self.gamma_group_name2inx = {} # keyed by gamma name, valued by group inx (default -1)
    self.gamma_duplicate_name2inx = {}
    self.gamma_duplicate_inx2names = []
    for name in self.gamma_structures.keys():
      self.gamma_group_name2inx[name] = -1
      self.gamma_duplicate_name2inx[name] = -1


    for name in self.nodes_name:
      node = self.nodes[name]
      # check input
      for i in range(len(node.input_gammas)):
        # border case
        if len(node.input_gammas) == 1 and node.input_gammas[0].gamma_mode == 'AllSelect':
          break
        input_gamma = node.input_gammas[i]

        pre_node = self.nodes[node.input_nodes[i]]
        inx = pre_node.output_nodes.index(name)
        pre_gamma = pre_node.output_gammas[inx]
        if pre_gamma.name != input_gamma.name:
          if pre_gamma.gamma_mode == 'AllSelect' \
              or input_gamma.gamma_mode == 'AllSelect':
            continue
          else:
            if self.gamma_duplicate_name2inx[pre_gamma.name] != -1:
              inx = self.gamma_duplicate_name2inx[pre_gamma.name]
              self.gamma_duplicate_name2inx[input_gamma.name] = inx
              self.gamma_duplicate_inx2names[inx].add(input_gamma.name)
            elif self.gamma_duplicate_name2inx[input_gamma.name] != -1:
              inx = self.gamma_duplicate_name2inx[input_gamma.name]
              self.gamma_duplicate_name2inx[pre_gamma.name] = inx
              self.gamma_duplicate_inx2names[inx].add(pre_gamma.name)
            else:
              cnt = len(self.gamma_duplicate_inx2names)
              self.gamma_duplicate_name2inx[pre_gamma.name] = cnt
              self.gamma_duplicate_name2inx[input_gamma.name] = cnt
              self.gamma_duplicate_inx2names.append( \
                      set([pre_gamma.name, input_gamma.name]))

      # check output
      for i in range(len(node.output_gammas)):
        # border case
        if len(node.output_gammas) == 1 and node.output_gammas[0].gamma_mode == 'AllSelect':
          break
        output_gamma = node.output_gammas[i]
        post_node = self.nodes[node.output_nodes[i]]
        inx = post_node.input_nodes.index(name)
        post_gamma = post_node.input_gammas[inx]
        if post_gamma.name != output_gamma.name:
          if post_gamma.gamma_mode == 'AllSelect' \
              or output_gamma.gamma_mode == 'AllSelect':
            continue
          else:
            if self.gamma_duplicate_name2inx[post_gamma.name] != -1:
              inx = self.gamma_duplicate_name2inx[post_gamma.name]
              self.gamma_duplicate_name2inx[output_gamma.name] = inx
              self.gamma_duplicate_inx2names[inx].add(output_gamma.name)
            elif self.gamma_duplicate_name2inx[output_gamma.name] != -1:
              inx = self.gamma_duplicate_name2inx[output_gamma.name]
              self.gamma_duplicate_name2inx[post_gamma.name] = inx
              self.gamma_duplicate_inx2names[inx].add(post_gamma.name)
            else:
              cnt = len(self.gamma_duplicate_inx2names)
              self.gamma_duplicate_name2inx[post_gamma.name] = cnt
              self.gamma_duplicate_name2inx[output_gamma.name] = cnt
              self.gamma_duplicate_inx2names.append( \
                    set([post_gamma.name, output_gamma.name]))

    # group
    self.gamma_group_inx2names = []
    group_cnt = 0
    for key in self.nodes_name:
      node = self.nodes[key]
      if node.op_type == 'Add':
        corr_gamma_structures = []
        corr_gamma_structures.extend(node.input_gammas)
        corr_gamma_structures.extend(node.output_gammas)
        for i in node.output_nodes:
          post_node = self.nodes[i]
          inx = post_node.input_nodes.index(node.name)
          gamma = post_node.input_gammas[inx]
          corr_gamma_structures.append(gamma)
          #corr_gamma_structures.extend(post_node.output_gammas)
        for i in node.input_nodes:
          pre_node = self.nodes[i]
          inx = pre_node.output_nodes.index(node.name)
          gamma = pre_node.output_gammas[inx]
          corr_gamma_structures.append(gamma)
        group_cnt = self._aggregate_corr_gammas( \
                            corr_gamma_structures,
                            group_cnt)

      elif node.op_type in _KEY_OPS_SINGLE_OUTPUT \
          and len(node.output_gammas) > 1:
        corr_gamma_structures = []
        corr_gamma_structures.extend(node.output_gammas)
        for i in node.output_nodes:
          post_node = self.nodes[i]
          inx = post_node.input_nodes.index(node.name)
          gamma = post_node.input_gammas[inx]
          corr_gamma_structures.append(gamma)
        group_cnt = self._aggregate_corr_gammas( \
                            corr_gamma_structures,
                            group_cnt)
      elif (node.op_type in _SPLIT_OPS \
          or node.op_type in _CONCAT_OPS) \
          and node.variables['axis'] not in _CHANNEL_AXIS:
        corr_gamma_structures = []
        corr_gamma_structures.extend(node.input_gammas)
        corr_gamma_structures.extend(node.output_gammas)
        for i in node.output_nodes:
          post_node = self.nodes[i]
          inx = post_node.input_nodes.index(node.name)
          gamma = post_node.input_gammas[inx]
          corr_gamma_structures.append(gamma)
        for i in node.input_nodes:
          pre_node = self.nodes[i]
          inx = pre_node.output_nodes.index(node.name)
          gamma = pre_node.output_gammas[inx]
          corr_gamma_structures.append(gamma)
        group_cnt = self._aggregate_corr_gammas( \
                            corr_gamma_structures,
                            group_cnt)

  def _aggregate_corr_gammas(self, corr_gamma_structures, group_cnt):
    """ Process the current found correlated gammas and aggregate them into the group records. """
    if len(corr_gamma_structures) <= 1:
      return group_cnt

    corr_gammas = []
    for gamma_structure in corr_gamma_structures:
      corr_gammas.append(gamma_structure.name)

    group_indices = set()
    for gamma in corr_gammas:
      group_indices.add(self.gamma_group_name2inx[gamma])
    group_indices = list(group_indices)

    if len(group_indices) == 1:
      if group_indices[0] == -1:
        self.gamma_group_inx2names.append(set())
        for gamma in corr_gammas:
          self.gamma_group_inx2names[group_cnt].add(gamma)
          self.gamma_group_name2inx[gamma] = group_cnt
        group_cnt += 1
    elif len(group_indices) == 2:
      if group_indices[0] == -1:
        group_inx = group_indices[1]
      elif group_indices[1] == -1:
        group_inx = group_indices[0]
      else:
        assert 'two many groups' # todo: description
      for gamma in corr_gammas:
        self.gamma_group_inx2names[group_inx].add(gamma)
        self.gamma_group_name2inx[gamma] = group_inx
    else:
      assert 'two many groups' # todo: description
    return group_cnt


  def _extract_gammas(self):
    """ Extract the input and output gammas for each node. """
    # add direction
    self._extract_add_directions()

    # BN
    explored = []
    candidate = {} # keyed by node name, valued by a dict with keys (`input_tag', `output_tag')
    for key in self.nodes_name[:]:
      node = self.nodes[key]
      if node.op_type == 'BN':
        candidate[key] = {} # for pop
        gamma = GammaStructure(node.variables_name['gamma'])
        self.gamma_structures[node.variables_name['gamma']] = gamma
        for i in node.input_nodes:
          node.add_input_gammas(gamma)
        for i in node.output_nodes:
          node.add_output_gammas(gamma)
        self._update_explored_state(key, explored, candidate)

    # process the beginning and ending node
    roots = []
    ending = []
    thrown = []
    for key in self.nodes_name[:]:
      node = self.nodes[key]
      if not node.input_nodes:
        if not node.output_nodes:
          self.nodes.pop(key)
          self.nodes_name.remove(key)
          thrown.append(key)
        else:
          roots.append(key)
      elif not node.output_nodes:
        ending.append(key)

    # roots
    for key in roots:
      if self.nodes[key].op_type in ['Conv2D', 'MatMul'] \
          or self.nodes[key].op_type in _SPLIT_OPS:
        gamma = GammaStructure( \
                    key+':input',
                    'AllSelect',
                    shape=self.nodes[key].input_size)
        self.gamma_structures[key+':input'] = gamma
        self.nodes[key].add_input_gammas(gamma)
        candidate[key] = {}
        candidate[key]['input_tag'] = {}
        candidate[key]['output_tag'] = {}
        for ii in self.nodes[key].output_nodes:
          candidate[key]['output_tag'][ii] = (ii in explored)
      elif self.nodes[key].op_type == 'BN':
        continue
      else:
        candidate[key] = {} # for pop
        gamma = GammaStructure( \
                    key+':output',
                    'AllSelect',
                    shape=self.nodes[key].output_size)
        self.gamma_structures[key+':output'] = gamma
        for i in self.nodes[key].output_nodes:
          self.nodes[key].add_output_gammas(gamma)
        self.gamma_structures[key+':input'] = gamma
        for i in self.nodes[key].input_nodes:
          self.nodes[key].add_input_gammas(gamma)
        self._update_explored_state(key, explored, candidate)

    # ending
    for key in ending:
      if self.nodes[key].op_type in ['Conv2D', 'MatMul'] \
          or self.nodes[key].op_type == 'Add' \
          or self.nodes[key].op_type in _CONCAT_OPS:
        gamma = GammaStructure( \
                    key+':output',
                    'AllSelect',
                    shape=self.nodes[key].output_size)
        self.gamma_structures[key+':output'] = gamma
        self.nodes[key].add_output_gammas(gamma)
        candidate[key] = {}
        candidate[key]['input_tag'] = {}
        candidate[key]['output_tag'] = {}
        for ii in self.nodes[key].input_nodes:
          candidate[key]['input_tag'][ii] = (ii in explored)
      elif self.nodes[key].op_type == 'BN':
        continue
      else:
        candidate[key] = {} # for pop
        gamma = GammaStructure( \
                    key+':input',
                    'AllSelect',
                    shape=self.nodes[key].output_size)
        self.gamma_structures[key+':input'] = gamma
        for i in self.nodes[key].input_nodes:
          self.nodes[key].add_input_gammas(gamma)
        self.gamma_structures[key+':output'] = gamma
        for i in self.nodes[key].output_nodes:
          self.nodes[key].add_output_gammas(gamma)
        self._update_explored_state(key, explored, candidate)

    self.roots = roots
    self.ending = ending
    self.thrown = thrown

    # BFS traversal
    candidate_backup = []
    next_explored = {}
    while candidate:
      # if there is no update, break
      if candidate == candidate_backup:
        if next_explored:
          # for conv2d with only one BN
          for key, direction in next_explored.items():
            self._extract_gammas_conv2d(key, direction)
            self._update_explored_state(key, explored, candidate)
        else:
          break
      candidate_backup = copy.deepcopy(candidate)
      next_explored = {}
      for key, value in candidate.items():
        node = self.nodes[key]
        # check input_tag
        input_explored = np.sum(value['input_tag'].values())
        input_total = len(node.input_nodes)
        input_all_tag = input_explored == input_total
        # check output_tag
        output_explored = np.sum(value['output_tag'].values())
        output_total = len(node.output_nodes)
        output_all_tag = output_explored == output_total
        if input_all_tag and output_all_tag:
          next_explored[key] = 'in_out'
        elif input_all_tag:
          next_explored[key] = 'input'
        elif output_all_tag:
          next_explored[key] = 'output'

      for key, direction in next_explored.items():
        # process the BN
        if self.nodes[key].op_type in ['Conv2D', 'MatMul']:
          # refine the direction for Conv2D
          if direction != 'in_out':
            continue
          self._extract_gammas_conv2d(key)
        elif self.nodes[key].op_type == 'Add':
          if direction not in self.add_directions[key]:
            continue
          self._extract_gammas_add(key, direction)
        elif self.nodes[key].op_type in _KEY_OPS_NONCHANNEL:
          self._extract_gammas_nonchannel(key, direction)
        elif self.nodes[key].op_type in _CONCAT_OPS:
          self._extract_gammas_concat(key, direction)
        elif self.nodes[key].op_type in _SPLIT_OPS:
          self._extract_gammas_split(key, direction)
        # todo: Matmul

        self._update_explored_state(key, explored, candidate)

  def _update_explored_state(self, key, explored, candidate):
    """ Update the explored nodes state. """
    # update
    explored.append(key)
    node = self.nodes[key]
    for i in node.input_nodes:
      if i not in explored:
        if i not in candidate.keys():
          candidate[i] = {}
          candidate[i]['input_tag'] = {}
          candidate[i]['output_tag'] = {}
          for ii in self.nodes[i].input_nodes:
            candidate[i]['input_tag'][ii] = False
          for ii in self.nodes[i].output_nodes:
            candidate[i]['output_tag'][ii] = False
        candidate[i]['output_tag'][key] = True
    for i in node.output_nodes:
      if i not in explored:
        if i not in candidate.keys():
          candidate[i] = {}
          candidate[i]['input_tag'] = {}
          candidate[i]['output_tag'] = {}
          for ii in self.nodes[i].input_nodes:
            candidate[i]['input_tag'][ii] = False
          for ii in self.nodes[i].output_nodes:
            candidate[i]['output_tag'][ii] = False
        candidate[i]['input_tag'][key] = True
    candidate.pop(key)


  def _extract_gammas_conv2d(self, name, direction='in_out'):
    """ Extract the input and output gammas for `Conv2D'. """
    node = self.nodes[name]
    input_nodes = node.input_nodes
    output_nodes = node.output_nodes
    if direction == 'input' or direction == 'in_out':
      if name not in self.roots:
        for i in input_nodes:
          pre_node = self.nodes[i]
          inx = pre_node.output_nodes.index(name)
          gamma = pre_node.output_gammas[inx]
          node.add_input_gammas(gamma)
      if direction == 'input':
        gamma = GammaStructure( \
                  name+':output',
                  'AllSelect',
                  shape=node.output_size)
        self.gamma_structures[name+':output'] = gamma
        for i in output_nodes:
          node.add_output_gammas(gamma)
    if direction == 'output' or direction == 'in_out':
      if name not in self.ending:
        for i in output_nodes:
          post_node = self.nodes[i]
          inx = post_node.input_nodes.index(name)
          gamma = post_node.input_gammas[inx]
          node.add_output_gammas(gamma)
      if direction == 'output':
        gamma = GammaStructure( \
                  name+':input',
                  'AllSelect',
                  shape=node.input_size)
        self.gamma_structures[name+':input'] = gamma
        for i in input_nodes:
          node.add_input_gammas(gamma)

  def _extract_gammas_nonchannel(self, name, direction):
    """ Extract the input and output gammas for opertions in _KEY_OPS_NONCHANNEL. """
    node = self.nodes[name]
    input_nodes = node.input_nodes
    output_nodes = node.output_nodes
    assert len(input_nodes) <= 1, \
        'number of the inputs for non-channel op is not one'
    if direction == 'input':
      for i in input_nodes:
        pre_node = self.nodes[i]
        inx = pre_node.output_nodes.index(name)
        gamma = pre_node.output_gammas[inx]
        node.add_input_gammas(gamma)
      for i in output_nodes:
        node.add_output_gammas(node.input_gammas[0])
    else:
      for i in output_nodes:
        post_node = self.nodes[i]
        inx = post_node.input_nodes.index(name)
        gamma = post_node.input_gammas[inx]
        node.add_output_gammas(gamma)
      for i in input_nodes:
        # only add `one'
        node.add_input_gammas(node.output_gammas[0])

  def _extract_gammas_add(self, name, direction):
    """ Extract the input and output gammas for `Add'. """
    node = self.nodes[name]
    input_nodes = node.input_nodes
    output_nodes = node.output_nodes
    assert len(input_nodes) <= 2, 'number of the inputs for add is not 2'
    if direction == 'input':
      for i in input_nodes:
        pre_node = self.nodes[i]
        inx = pre_node.output_nodes.index(name)
        gamma = pre_node.output_gammas[inx]
        node.add_input_gammas(gamma)
      for i in output_nodes:
        node.add_output_gammas(node.input_gammas[0])
    elif direction == 'output':
      for i in output_nodes:
        post_node = self.nodes[i]
        inx = post_node.input_nodes.index(name)
        gamma = post_node.input_gammas[inx]
        node.add_output_gammas(gamma)
      for i in input_nodes:
        node.add_input_gammas(node.output_gammas[0])
    else:
      for i in input_nodes:
        pre_node = self.nodes[i]
        inx = pre_node.output_nodes.index(name)
        gamma = pre_node.output_gammas[inx]
        node.add_input_gammas(gamma)
      for i in output_nodes:
        post_node = self.nodes[i]
        inx = post_node.input_nodes.index(name)
        gamma = post_node.input_gammas[inx]
        node.add_output_gammas(gamma)

  def _extract_gammas_concat(self, name, direction):
    """ Extract the input and output gammas for operations in _CONCAT_OPS. """
    node = self.nodes[name]
    input_nodes = node.input_nodes
    output_nodes = node.output_nodes
    if direction == 'input':
      # input gammas
      indices = [0]
      if name not in self.roots:
        for i in input_nodes:
          pre_node = self.nodes[i]
          if pre_node.op_type in _SPLIT_OPS:
            inx = pre_node.output_nodes.index(name)
          else:
            inx = 0
          gamma = pre_node.output_gammas[inx]
          node.add_input_gammas(gamma)
          indices.append(indices[-1]+pre_node.output_size[inx])
      # output gammas
      if node.variables['axis'] in _CHANNEL_AXIS:
        gamma_output_name = name + ':gamma'
        gamma_output = GammaStructure(gamma_output_name, 'Concat')
        gamma_output.set_indices(indices)
        self.gamma_structures[gamma_output_name] = gamma_output
        for gamma in node.input_gammas:
          gamma_output.add_corr_gammas(gamma)
      else:
        gamma_output = node.input_gammas[0]
      for i in output_nodes:
        node.add_output_gammas(gamma_output)
    else:
      # output gammas
      if name in self.ending:
        gamma = node.output_gammas[0]
      else:
        for i in output_nodes:
          post_node = self.nodes[i]
          inx = post_node.input_nodes.index(name)
          gamma = post_node.input_gammas[inx]
          node.add_output_gammas(gamma)
      # input gammas
      if node.variables['axis'] in _CHANNEL_AXIS:
        start_inx = 0
        for i in range(len(input_nodes)):
          end_inx = start_inx + node.input_size[i]
          sub_gamma = GammaStructure( \
                        gamma.name+":%d"%i,
                        'Split',
                        start_inx,
                        end_inx)
          sub_gamma.add_corr_gammas(gamma)
          self.gamma_structures[gamma.name+":%d"%i] = sub_gamma
          node.add_input_gammas(sub_gamma)
          start_inx = end_inx
      else:
        gamma = node.output_gammas[0]
        for i in input_nodes:
          node.add_input_gammas(gamma)
      return

  def _extract_gammas_split(self, name, direction):
    """ Extract the input and output gammas for operations in _SPLIT_OPS. """
    node = self.nodes[name]
    input_nodes = node.input_nodes
    output_nodes = node.output_nodes
    assert len(input_nodes) <= 1, 'number of the inputs for split is not one'
    if direction == 'output':
      # output gammas
      indices = [0]
      for i in output_nodes:
        post_node = self.nodes[i]
        inx = post_node.input_nodes.index(name)
        gamma = post_node.input_gammas[inx]
        node.add_output_gammas(gamma)
        indices.append(indices[-1]+post_node.input_size[inx])
      # input gammas
      if node.variables['axis'] in _CHANNEL_AXIS:
        gamma_input_name = name + ':gamma'
        gamma_input = GammaStructure(gamma_input_name, 'Concat')
        gamma_input.set_indices(indices)
        self.gamma_structures[gamma_input_name] = gamma_input
        for gamma in node.output_gammas:
          gamma_input.add_corr_gammas(gamma)
      else:
        gamma_input = node.output_gammas[0]
      for i in input_nodes:
        node.add_input_gammas(gamma_input)
    else:
      # input gammas
      for i in input_nodes:
        pre_node = self.nodes[i]
        inx = pre_node.output_nodes.index(name)
        gamma = pre_node.output_gammas[inx]
        node.add_input_gammas(gamma)
      # output gammas
      if node.variables['axis'] in _CHANNEL_AXIS:
        start_inx = 0
        for i in range(len(output_nodes)):
          end_inx = start_inx + node.output_size[i]
          sub_gamma = GammaStructure( \
                        gamma.name+":%d"%i,
                        'Split',
                        start_inx,
                        end_inx)
          sub_gamma.add_corr_gammas(gamma)
          self.gamma_structures[gamma.name+':%d'%i] = sub_gamma
          node.add_output_gammas(sub_gamma)
          start_inx = end_inx
      else:
        gamma = node.input_gammas[0]
        for i in output_nodes:
          node.add_output_gammas(gamma)


class PatternMatch(object):
  """Pattern detection based on given pattern (mainly for unfused BN)."""
  def __init__(self, pattern_map, entry_op_check, first_key, numinputs, numoutputs):
    self.matched_bns = []
    self.matched_first_keys = []
    self.numinputs = numinputs
    self.numoutputs = numoutputs
    self.opinputs = [-1 for i in range(self.numinputs)]
    self.opoutputs = [-1 for i in range(self.numoutputs)]
    self.nodes = {}
    self._num_matched = 0
    self.pattern_map = pattern_map
    self.entry_op_check = entry_op_check
    self.first_key = first_key

  def checkinputs(self, node, pattern_node):
    """ Check inputs for the current node whether it corresponds to the pattern. """
    for i, t in enumerate(node.inputs):
      input_key = pattern_node["inputs"][i]
      if isinstance(input_key, int):
        self.opinputs[input_key] = t
      elif (t.op.node_def.op == self.pattern_map[input_key]["Op"] \
          and t.op not in self.matched_first_keys):
        if input_key not in self.nodes.keys():
          self.nodes[input_key] = [t.op, False]
      else:
        return False
    return True

  def checkoutputs(self, node, pattern_node):
    """ Check outputs for the current node whether it corresponds to the pattern. """
    for i, t in enumerate(node.outputs):
      output_keys = pattern_node["outputs"][i]
      t_consumers = []
      for c in t.consumers():
        if 'gradients' in c.name:
          continue
        t_consumers.append(c)

      outgoing_tensor = False
      for output_key in output_keys:
        if isinstance(output_key, int):
          outgoing_tensor = True
          self.opoutputs[output_key] = t
        else:
          found = False
          for j, c in enumerate(t_consumers):
            if (c.node_def.op == self.pattern_map[output_key]["Op"] \
                and c not in self.matched_first_keys):
              found = True
              if output_key not in self.nodes.keys():
                self.nodes[output_key] = [c, False]
              break
          if not found:
            return False
      if (not outgoing_tensor and len(t_consumers) > len(output_keys)):
        return False
    return True

  def visit_nodes(self):
    """ Check whether all the nodes in the pattern are visited. """
    all_visited = False
    while not all_visited:
      all_visited = True
      for key, n in self.nodes.items():
        if not n[1]:
          all_visited = False
          pattern_node = self.pattern_map[key]
          if not self.checkinputs(n[0], pattern_node):
            return False
          if not self.checkoutputs(n[0], pattern_node):
            return False
          n[1] = True
    return True


  def match(self, graph):
    """ Function to match the batchnomr pattern in the given graph `graph' """
    first_pattern_node = self.pattern_map[self.first_key]
    for node in graph.get_operations():
      #if node.node_def.op == first_pattern_node["Op"] \
      if self.entry_op_check(node) \
          and node not in self.matched_first_keys:
        self.opinputs = [-1 for i in range(self.numinputs)]
        self.opoutputs = [-1 for i in range(self.numoutputs)]
        self.nodes = {self.first_key: [node, True]}
        if not self.checkinputs(node, first_pattern_node):
          continue
        if not self.checkoutputs(node, first_pattern_node):
          continue
        if not self.visit_nodes():
          continue

        self._num_matched = self._num_matched + 1

        # add found BN to `matched_nodes'
        self.matched_bns.append(self.nodes)

        for n in self.nodes.values():
          self.matched_first_keys.append(n[0])

    return self.matched_bns

class BatchNormPattern(object):
  def __init__(self):
    self.first_key = "rsqrt"
    self.numinputs = 4
    self.numoutputs = 2
    self.pattern_map = {\
         "rsqrt": \
              {"Op": "Rsqrt",
               "inputs": [0], #input tensor ids of the fused op
               "outputs": [["mul"]],
              },
         "mul": \
              {"Op": "Mul",
               "inputs": ['rsqrt', 1],
               "outputs": [["mul_1", 0]]
              },
         "mul_1": \
              {"Op": "Mul",
               "inputs": [2, "mul"],
               "outputs": [["add_1"]],
              },
         "add_1": \
              {"Op": "Add",
               "inputs": ["mul_1", 3],
               "outputs": [[1]],
              }}

  def entry_op_check(self, op):
    return op.type == self.pattern_map[self.first_key]['Op']


class TensorDotPattern(object):
  def __init__(self):
    self.first_key = "matmul"
    self.numinputs = 7
    self.numoutputs = 1
    self.pattern_map = {\
         "transpose": \
              {"Op": "Transpose",
               "inputs": [0, 1], #input tensor ids of the fused op
               "outputs": [["reshape"]],
              },
         "transpose_1": \
              {"Op": "Transpose",
               "inputs": [2, 3], #input tensor ids of the fused op
               "outputs": [["reshape_1"]],
              },
         "reshape": \
              {"Op": "Reshape",
               "inputs": ['transpose', 4],
               "outputs": [["matmul"]]
              },
         "reshape_1": \
              {"Op": "Reshape",
               "inputs": ['transpose_1', 5],
               "outputs": [["matmul"]]
              },
         "matmul": \
              {"Op": "MatMul",
               "inputs": ["reshape", "reshape_1"],
               "outputs": [["reshape_2"]],
              },
         "reshape_2": \
              {"Op": "Reshape",
               "inputs": ["matmul", 6],
               "outputs": [[0]],
              }}

  def entry_op_check(self, op):
    return op.type == self.pattern_map[self.first_key]['Op'] \
        and 'Tensordot' in op.name
