import os
import sys
import pdb
import time
import math
import multiprocessing
import tensorflow as tf
import numpy as np

from numpy import linalg as LA
from collections import OrderedDict
from enum import Enum, unique
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import constant_op
from tensorflow.core.framework import attr_value_pb2

from amp_util import *

calibration_num_proc = 16
root_dir = os.path.abspath(os.path.dirname(__file__))

@unique
class OfflineQuantizeMethod(Enum):
  KL = 1
  MAX = 2
  MAXC = 3

method_map = {'KL': OfflineQuantizeMethod.KL,
              'MAX': OfflineQuantizeMethod.MAX,
              'MAXC': OfflineQuantizeMethod.MAXC}

def quantize_weights(graph_def, kernel_dict, num_bits=None):
  # per-channel quantization
  kernel_list = list()
  scale_int_dict = dict()
  for kernel_name in kernel_dict:
      kernel_list.append(kernel_name)
      scale_int_dict[kernel_name] = pow(2, num_bits[kernel_name]) / 2 - 1

  for i, node in enumerate(graph_def.node):
      if (node.op == 'Const') and (node.name in kernel_list):
          #weight_name = node.name
          weights = tensor_util.MakeNdarray(node.attr['value'].tensor)
          weights_shape = weights.shape
          #weights_ori = weights.copy()
          weights = weights.reshape((-1, weights_shape[-1]))
          abs_max_value = np.amax(np.abs(weights), axis=0).astype(np.float32)
          scale_factor = float(scale_int_dict[node.name]) / abs_max_value
          if len(weights_shape) == 2:
              scale_factor = scale_factor.reshape((1, weights_shape[-1]))
          elif len(weights_shape) == 4:
              scale_factor = scale_factor.reshape((1, 1, 1, weights_shape[-1]))
          #quant_weight = np.round(weights_ori * scale_factor) / scale_factor
          #node.MergeFrom(constant_op.constant(quant_weight,
            #             dtype=tf.float32, name=node.name).op.node_def)
          #node.name = weight_name
          if node.name in kernel_dict:
              kernel_dict[node.name] = (scale_factor, num_bits[node.name])
  return kernel_dict


def computeScaleValue(data_list, method=OfflineQuantizeMethod.KL, name='',
                      num_bits=8, bin_num=2048):
  max_range = float(pow(2, num_bits) // 2 - 1)
  quant_num = int(pow(2, num_bits) // 2)

  flatten_abs_data = np.abs(np.concatenate(data_list).ravel())
  data_max = np.max(flatten_abs_data)
  is_one_hot = np.all(np.add(flatten_abs_data==0, flatten_abs_data==1))
  # max, per-tensor for activation
  if (method == OfflineQuantizeMethod.MAX or is_one_hot) and (num_bits < 8):
    data_max_list = [np.max(np.abs(data_array)) for data_array in data_list]
    data_max = np.mean(np.array(data_max_list))
    max_scale = max_range / (data_max+sys.float_info.epsilon)
    return name, (max_scale, num_bits)
  # KL, per-tensor for activation
  bin_size = float(data_max) / bin_num
  hist = np.zeros((bin_num))
  for data in data_list:
    data = np.abs(data)
    data = np.int32(np.minimum(np.floor(data / bin_size), bin_num-1))
    tmp_hist = np.bincount(data[np.where(data > 0)])
    hist[:tmp_hist.shape[0]] += tmp_hist

  start_idx = np.where(hist>0)[0][0]
  start_idx = min(max(start_idx+1, quant_num), bin_num)
  KL = np.ones((bin_num)) * float('inf')
  for i in range(start_idx, bin_num+1):
    P = hist[:i].copy()
    P[i-1] += np.sum(hist[i:])
    P /= sum(P)

    # Need to optimize
    # For an array of length l that should be splitinto n sections,
    # it returns l % n sub-arrays of size l//n + 1 and the rest of size l//n.
    tmp_hist = hist[:i].copy()
    sub_1_num = i % quant_num
    sub_1_len = i // quant_num + 1
    sub_2_num = quant_num - sub_1_num
    sub_2_len = i // quant_num
    Q_array_1 = tmp_hist[:sub_1_num*sub_1_len].reshape(sub_1_num, sub_1_len)
    Q_array_2 = tmp_hist[sub_1_num*sub_1_len:].reshape(sub_2_num, sub_2_len)
    Q_quant_1 = np.sum(Q_array_1, 1)
    Q_quant_2 = np.sum(Q_array_2, 1)
    Q_quant = np.concatenate((Q_quant_1, Q_quant_2))
    Q_quant_num_1 = np.sum(Q_array_1 > 0, 1)
    Q_quant_num_2 = np.sum(Q_array_2 > 0, 1)
    Q_quant_num = np.int32(np.concatenate((Q_quant_num_1, Q_quant_num_2)))
    tmp_range_1 = np.array(range(sub_1_num)).reshape(-1, 1)
    tmp_range_2 = (np.array(range(sub_2_num)) + sub_1_num).reshape(-1, 1)
    Q_quant_idx_1 = (np.ones(Q_array_1.shape) * tmp_range_1).reshape(-1)
    Q_quant_idx_2 = (np.ones(Q_array_2.shape) * tmp_range_2).reshape(-1)
    Q_quant_idx = np.int32(np.concatenate((Q_quant_idx_1, Q_quant_idx_2)))

    Q = np.zeros((i))
    tmp_idx = np.where(hist[:i] > 0)
    Q[tmp_idx] = Q_quant[Q_quant_idx[tmp_idx]] / Q_quant_num[Q_quant_idx[tmp_idx]]
    Q /= sum(Q)

    tmp_idx = np.where(Q == 0)
    Q[tmp_idx] = sys.float_info.epsilon
    tmp_idx = np.where(P > 0)
    KL[i-1] = np.sum(P[tmp_idx] * np.log(P[tmp_idx] / Q[tmp_idx]))

  m = np.argmin(KL)
  threshold = (m + 0.5) * bin_size
  scale_kl = max_range / threshold
  return name, (scale_kl, num_bits)


def computeAllOfflineScale(featmap_name, featmap_data,
                           method=OfflineQuantizeMethod.KL,
                           num_bits=None):
  scale_dict = dict()
  for i in range(len(featmap_name)):
    _, scale_bits = computeScaleValue(featmap_data[featmap_name[i]], method,
                                featmap_name[i], num_bits[featmap_name[i]])
    scale_dict[featmap_name[i]] = scale_bits
  return scale_dict


def computeAllOfflineScaleMultiProc(featmap_name, featmap_data,
                                    method=OfflineQuantizeMethod.KL,
                                    process_num=calibration_num_proc,
                                    num_bits=None):
  # offline KL
  scale_dict = dict()
  results = list()
  pool = multiprocessing.Pool(processes=min(len(featmap_name), process_num))
  for i in xrange(len(featmap_name)):
    results.append(pool.apply_async(
        computeScaleValue,
        (featmap_data[featmap_name[i]], method, featmap_name[i], num_bits[featmap_name[i]])))
  pool.close()
  pool.join()
  for result in results:
    scale_name, scale_bits = result.get()
    scale_dict[scale_name] = scale_bits
  return scale_dict


def compute_scales(graph_def, kernel_dict, act_dict, method='KL', num_bits=None):
    # kernel, per-channel max
    kernel_dict = quantize_weights(graph_def, kernel_dict, num_bits=num_bits[1])
    # activation
    method = method_map[method]
    featmap_name = list()
    for act_name in act_dict:
        featmap_name.append(act_name)

    act_dict = computeAllOfflineScaleMultiProc(featmap_name, act_dict, method=method, num_bits=num_bits[0])
    return act_dict, kernel_dict


def quantize_dequantize(graph_def, node, input_node_name,
                        scale_factor, name='act'):
    data_type = tf.float32 #tf.DType(node.attr['T'].type)
    # act scale
    const_node = graph_def.node.add()
    const_node.op = 'Const'
    const_node.name = node.name+'_%s_scale' % name
    const_node.MergeFrom(constant_op.constant(scale_factor,
                         dtype=tf.float32, name=const_node.name).op.node_def)
    # tf.multiply
    multi_node = graph_def.node.add()
    multi_node.op = 'Mul'
    multi_node.name = node.name+'_%s_mul' % name
    multi_node.input.extend([input_node_name, const_node.name])
    multi_node.attr['T'].CopyFrom(tf.AttrValue(type=data_type.as_datatype_enum))
    # tf.round
    round_node = graph_def.node.add()
    round_node.op = 'Round'
    round_node.name = node.name+'_%s_round' % name
    round_node.input.extend([multi_node.name])
    round_node.attr['T'].CopyFrom(tf.AttrValue(type=data_type.as_datatype_enum))
    # tf.divide
    div_node = graph_def.node.add()
    div_node.op = 'RealDiv'
    div_node.name = node.name+'_%s_div' % name
    div_node.input.extend([round_node.name, const_node.name])
    div_node.attr['T'].CopyFrom(tf.AttrValue(type=data_type.as_datatype_enum))
    if name == 'act':
        node.input[0] = div_node.name
    elif name == 'weight':
        node.input[1] = div_node.name
    return graph_def, node


def graph_fake_quantize(graph_def, kernel_dict, act_dict):
    for i, node in enumerate(graph_def.node):
        if (node.op in ['Conv2D', 'MatMul']):
            input_act_node = node.input[0]
            input_weight_node = node.input[1]

            if 'read' in input_weight_node:
                weight_const = input_weight_node[:-5]
            else:
                weight_const = input_weight_node

            act_scale = act_dict[input_act_node+':0'][0]
            weight_scale = kernel_dict[weight_const][0]
            # quantize-dequantize
            graph_def, node = quantize_dequantize(graph_def, node, input_act_node,
                                            act_scale, name='act')
            graph_def, node = quantize_dequantize(graph_def, node, input_weight_node,
                                            weight_scale, name='weight')
    return graph_def


def copyGraphDef(graph_def):
  new_graph_def = tf.GraphDef()
  new_graph_def.CopyFrom(graph_def)
  return new_graph_def

int8_acts = ['resnet_model/Pad:0',
             'resnet_model/Reshape:0']
int8_kernels = ['resnet_model/conv2d/kernel',
                'resnet_model/dense/kernel']

def kl_calib(graph_def, task, args, logging=None):
  # extrat information
  act_dict, kernel_dict, _, _ = extract_graph_node(graph_def, logging)
  # bits allocation
  act_bits, kernel_bits = dict(), dict()
  for act_name in act_dict:
      if act_name in int8_acts:
          act_bits[act_name] = 8
      else:
          act_bits[act_name] = args.bits

  for kernel_name in kernel_dict:
      if kernel_name in int8_kernels:
          kernel_bits[kernel_name] = 8
      else:
          kernel_bits[kernel_name] = args.bits

  # obtain scaling factors
  calib_path = os.path.join(root_dir, '../calib/')
  if not os.path.exists(calib_path):
      os.mkdir(calib_path)
  act_dict_file = os.path.join(calib_path, 'act_dict_%s.npy' % args.bits)

  if not os.path.isfile(act_dict_file):
      # dump activations
      act_dict = task.calib_dump(graph_def, act_dict=act_dict)
      # calculate the scaling factors, both activations and weights
      act_dict, kernel_dict = compute_scales(graph_def, kernel_dict, act_dict,
                                method=args.method, num_bits=[act_bits, kernel_bits])
      np.save(act_dict_file, act_dict)
  else:
      print('load the pre-calib: %s' % act_dict_file)
      act_dict = np.load(act_dict_file, allow_pickle=True).item()

  return
