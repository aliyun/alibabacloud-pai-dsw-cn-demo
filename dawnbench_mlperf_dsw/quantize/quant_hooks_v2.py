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
"""Some useful session run hooks."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import multiprocessing

import numpy as np

from enum import Enum, unique
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import training
from quantize.fake_quantize_v2 import quantize_graph

root_dir = os.path.abspath(os.path.dirname(__file__))

@unique
class OfflineQuantizeMethod(Enum):
  KL = 1
  MAX = 2
  PERC = 3


MODE_MAP = {1: OfflineQuantizeMethod.KL,
            2: OfflineQuantizeMethod.MAX,
            3: OfflineQuantizeMethod.PERC,}

calibration_num_proc = 16


def computeScaleValue(data_list, method=OfflineQuantizeMethod.KL, name='',
                      num_bits=8, bin_num=2048):
  max_range = float(pow(2, num_bits) // 2 - 1)
  quant_num = int(pow(2, num_bits) // 2)

  flatten_abs_data = np.abs(np.concatenate(data_list).ravel())
  data_max = np.max(flatten_abs_data)
  is_one_hot = np.all(np.add(flatten_abs_data==0, flatten_abs_data==1))
  # max-range, per-tensor for activation
  if (method == OfflineQuantizeMethod.MAX or is_one_hot) and (num_bits < 8):
    data_max_list = [np.max(np.abs(data_array)) for data_array in data_list]
    data_max = np.mean(np.array(data_max_list))
    max_scale = max_range / (data_max+sys.float_info.epsilon)
    return name, (max_scale, num_bits)
  # histgram of the activation
  bin_size = float(data_max) / bin_num
  hist = np.zeros((bin_num))
  for data in data_list:
    data = np.abs(data)
    data = np.int32(np.minimum(np.floor(data / bin_size), bin_num-1))
    tmp_hist = np.bincount(data[np.where(data > 0)])
    hist[:tmp_hist.shape[0]] += tmp_hist

  start_idx = np.where(hist>0)[0][0]
  start_idx = min(max(start_idx+1, quant_num), bin_num)
  # 99.999th percentile range value
  if (method == OfflineQuantizeMethod.PERC):
    perc_i = 0
    hist_total = np.sum(hist)
    for i in range(start_idx, bin_num+1):
      P = hist[:i].copy()
      P_total = np.sum(P)
      P_perc = P_total / float(hist_total)
      if (P_perc > 0.99999):
        perc_i = i-1
        break
    threshold = (perc_i + 0.5) * bin_size
    scale_perc = max_range / threshold
    return name, (scale_perc, num_bits)

  # KL-scale, per-tensor for activation
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


# pylint: disable=protected-access
class QuantHook(training.SessionRunHook):

  def __init__(self,
               calib_file='calib_224_64.npy',
               bits=8,
               input_name = 'IteratorGetNext:0',
               int8_layers = ['resnet_model/conv2d/Conv2D',
                              'resnet_model/dense/MatMul'],
               online=False,
               quant_copy_num=1,
               quant_mode=1,
               finish=False
               ):

    self._bits = bits
    self._input_name = input_name
    self._online = online and (not finish)
    self._int8_layers = int8_layers
    self._quant_copy_num = quant_copy_num
    self._quant_mode = MODE_MAP[quant_mode]
    self._quant_mode_t = quant_mode

    calib_path = os.path.join(root_dir, '../calib/')
    if not os.path.exists(calib_path):
        os.mkdir(calib_path)

    self._pre_calib = calib_path
    self._calib_file = os.path.join(calib_path, calib_file)

    if self._online:
      calib_data = np.load(self._calib_file, allow_pickle=True)
      self._calib_data = list()
      for i, calib_dict in enumerate(calib_data):
        self._calib_data.append({self._input_name: calib_dict.values()[0]})


  def begin(self):
    # graph rewrite
    self._act_dict = quantize_graph(bits=self._bits,
                                    pre_calib=self._pre_calib,
                                    int8_layers=self._int8_layers,
                                    quant_copy_num=self._quant_copy_num,
                                    online=self._online,
                                    method=self._quant_mode_t)

    if self._online:
      self._act_list = list()
      self._act_bits = dict()
      for act_name in self._act_dict:
        self._act_list.append(act_name)
        self._act_bits[act_name] = self._act_dict[act_name][1]


  def after_create_session(self, session, coord):  # pylint: disable=unused-argument

    if self._online:
      # obtain the calib data
      print('KL calib: obtain the calib data')
      act_dict = dict()
      for i, calib_dict in enumerate(self._calib_data):
        acts = session.run(self._act_list, feed_dict=calib_dict)
        for act_name, act in zip(self._act_list, acts):
          if not (act_name in act_dict):
            act_dict[act_name] = [act]
          else:
            act_dict[act_name].append(act)
      # calculate the KL scaling factor
      print('KL calib: calculate the KL scaling factor')
      act_dict = computeAllOfflineScaleMultiProc(featmap_name=self._act_list,
                                                 featmap_data=act_dict,
                                                 method=self._quant_mode,
                                                 num_bits=self._act_bits)
      act_dict_file = os.path.join(self._pre_calib, 'act_dict_%s.npy' % self._bits)
      print('Save the online-calib: %s' % act_dict_file)
      np.save(act_dict_file, act_dict)
