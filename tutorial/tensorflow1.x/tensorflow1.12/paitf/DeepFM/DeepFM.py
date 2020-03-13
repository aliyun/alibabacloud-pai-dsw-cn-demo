# coding:utf-8
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
"""A implenetion of DeepFM, fork from git@github.com:ChenglongChen/tensorflow-DeepFM.git"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import time, os, datetime
import tensorflow as tf

from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.lookup import KeyValueTensorInitializer
from tensorflow.contrib.lookup import HashTable
from tensorflow.python.training.session_run_hook import SessionRunHook

 
class Reader(object):
    """A cross platform Reader, OSS ODPS and localhost"""
    def __init__(self, path,
            cols, delimiter, task_index=0,
            worker_num=1):
        self.path = path
        self.cols = cols
        self.delimiter = delimiter
        self.task_index = task_index
        self.worker_num = worker_num

        if self.path.startswith("odps://"):
            self.reader = \
                    self.dataset_load_from_odps(
                self.path, cols,
                self.task_index, self.worker_num)
        else:
            if not isinstance(cols, list):
                cols = cols.split(",")
            is_index_cols = True 
            for c in cols:
                if not c.isdigit() or str(int(float(c))) != c:
                    is_index_cols = False
                    break
            header = False
            if not is_index_cols:
                f = tf.gfile.Open(self.path)
                head = f.readline()
                head = head.lstrip().rstrip()
                if len(head) <= 0:
                    raise Exception("csv file head process failed")
                csv_cols = head.split(self.delimiter)
                header = True
            else:
                csv_cols = cols

            try:
                cols_idx = [csv_cols.index(n) for n in cols] 
                self.reader = self.dataset_load_from_local_csv(
                        self.path, cols_idx, self.delimiter, header)
            except Exception as e:
                print("cols not find in table", str(e))
                raise e

    
    def read(self):
        return self.reader

    def dataset_load_from_local_csv(self, filenames, cols, delimiter, header=True):
        cols_len = len(cols)
        d = tf.data.experimental.CsvDataset(
                filenames = filenames, 
                record_defaults=[''] * cols_len,
                select_cols = [x for x in range(len(cols))],
                field_delim = delimiter,
                header = header
                )
        return d

    def dataset_load_from_odps(self,
        tablename, cols, 
            task_index, worker_num):

        cols = cols.split(",")
 
        d = tf.data.TableRecordDataset(
            [tablename], record_defaults=['']*len(cols),
            selected_cols = ",".join(cols),
            slice_id = task_index,
            slice_count = worker_num)
        return d

class Writer(object):
    """A cross platform Writer, OSS ODPS and localhost"""
    def __init__(self, path, incides, task_index):
        self.path = path.lstrip().rstrip()
        self.incides = incides
        self.task_index = task_index
        self.paltform = None

        if self.path.startswith("odps://"):
            self.writer = tf.python_io.TableWriter(self.path, self.task_index)
            def _write(record):
                self.writer.write(record, incides)
            self.write = _write
            self.platform = "odps"
        else:
            self.writer = open(self.path, "w")
            def _write(record):
                self.writer.writelines([",".join([str(r) for r in record])+"\n"])
            self.write = _write

    def write(self):
        raise Exception("unimplemented method")

    def close(self):
        self.writer.close()

   
class DeepFM(object):
    """DeepFM model structure class"""
    def __init__(self, 
         input_name, feature_cols, feature_max_size,
         kv_map,
         label_col_name, checkpoint_dir, output_name='', 
         kvs_delimiter=",",
         kv_delimiter=":",
         sync_type="async",
         multi_tags_col_name=None,
         multi_tags_max_size=100,
         embedding_size=8, dropout_fm=[1.0, 1.0],
         deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
         deep_layers_activation=tf.nn.relu,
         batch_size=256,
         learning_rate=0.001, optimizer_type="adam",
         adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8,
         adagrad_initial_accumulator_value=1e-8,
         momentum=0.95,
         batch_norm=False, batch_norm_decay=0.995,
         verbose=False, random_seed=2016,
         use_fm=True, use_deep=True,
         loss_type="logloss", l2_reg=0.0,
         epoch=1, num_steps=None, mode="train",
         server=None, cluster=None, task_index=0, worker_num=1,
         ):
        """Build DeepFM graph, support kv freatures and multi tags feature"""

        assert (use_fm or use_deep)
        assert loss_type in ["logloss", "mse"], \
            "loss_type can be either 'logloss' for classification task or 'mse' for regression task"
        assert sync_type in ["async", "sync"], \
                "sync_type must be either 'async' or 'sync'"

        self.features_config = feature_cols 
        if multi_tags_col_name:
            self.features_config += multi_tags_col_name
        self.feature_max_size = feature_max_size      # denote as M, size of the feature dictionary
        self.kv_map = kv_map
        self.feature_size = len(self.kv_map) // 2          # denote as F, size of the feature fields
        self.embedding_size = embedding_size          # denote as K, size of the feature embedding
        self.label_col_name = label_col_name
        self.multi_tags_col_name = multi_tags_col_name
        self.multi_tags_feature_size = len(multi_tags_col_name) if multi_tags_col_name else 0
        self.multi_tags_max_size = multi_tags_max_size
        self.multi_tags_embedding_size = embedding_size
        self.kvs_delimiter = kvs_delimiter
        self.kv_delimiter = kv_delimiter
        
        self.all_feature_size = self.feature_size + self.multi_tags_feature_size

        self.sync_type = sync_type
        self.sync_opt_hook = None
        self.feature_dict = {}

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.use_multi_tags = True if self.multi_tags_col_name else False
        self.l2_reg = l2_reg
        self.epoch = epoch
        self.num_steps = num_steps

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.adagrad_initial_accumulator_value = adagrad_initial_accumulator_value
        self.momentum = momentum

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.mode = mode

        self.server = server
        self.cluster = cluster
        self.task_index = task_index
        self.worker_num = worker_num
        self.input_name= input_name
        self.checkpoint_dir = checkpoint_dir
        self.output_name = output_name
        
        #build all graph
        self.build_graph_distibute_mode()
       
    def build_graph_distibute_mode(self):
        self.graph = tf.Graph()
        self.is_distibute = True if self.cluster else False

        def _build_graph():
            self._build_kv_map()
            self._input_fn()
            self._build_graph(serving_mode=False)
        
        if self.is_distibute:
            with self.graph.as_default() as g, g.device(tf.train.replica_device_setter(
                       worker_device='/job:worker/task:%d' % self.task_index,
                       cluster=self.cluster)):
                _build_graph()
        else:
            with self.graph.as_default() as g:
                _build_graph()

        if self.verbose > 0:
            print("build graph done")
    
    def get_tf_run_config(self):
        if self.is_distibute:
            config = tf.ConfigProto(device_filters=[
                    '/job:ps',
                    '/job:worker/task:%d' % self.task_index
                ])
        else:
            config = tf.ConfigProto()
        return config

    def get_master_target(self):
        return self.server.target if self.is_distibute else ''

    def _build_kv_map(self):
        kv_map_keys = []
        kv_map_vals = []
        for i in range(0, len(self.kv_map), 2):
            s, e  = self.kv_map[i+1]
            if e <=s :
                raise Exception("kv map param error, start: %d, end %d" %(s, e))
            kv_map_keys.append(tf.range(s, e))
            kv_map_vals.append(tf.tile(tf.constant([i//2]),[e-s]))
            print("kv map range start:",s," end:",e," solt:",i//2)

        kv_map_keys = tf.cast(tf.concat(kv_map_keys, axis=0), dtype=tf.int64)
        kv_map_vals = tf.cast(tf.concat(kv_map_vals, axis=0), dtype=tf.int64)

        it = KeyValueTensorInitializer(kv_map_keys, kv_map_vals)
        self.kv_map_t = HashTable(it, default_value=-1)


    def _multi_tags_input_fn(self, multi_tag_col):
        multi_tag_kv = multi_tag_col

        kvs = tf.string_split([multi_tag_kv], self.kvs_delimiter)
        kvs = tf.string_split(kvs.values, self.kv_delimiter)
        key_values = tf.reshape(kvs.values, kvs.dense_shape)
        idxs, vals = tf.split(key_values, num_or_size_splits=2, axis=1)

        idxs = tf.reshape(tf.string_to_number(idxs, out_type=tf.int32), [-1])
        vals = tf.reshape(vals, [-1])
        return idxs, vals
 
    def _multi_tags_embedding(self, multi_tag_vals, tag_max_size, tag_embedding_size):
        # split input multi tags string, like 1:0.9,2:0.jj7
        kvs = tf.string_split(multi_tag_vals, self.kvs_delimiter)
        default_value = "0" + self.kv_delimiter + "0"
        kvs = tf.sparse.to_dense(kvs, default_value=default_value)
        def map_split(kv):
            return tf.string_split(kv, self.kv_delimiter).values
        kvs = tf.map_fn(map_split, kvs)
        shape = tf.shape(kvs)
        # pick idxs and vals by reshape
        kvs = tf.reshape(kvs, [-1,2])
        idxs = kvs[:,0]
        idxs = tf.string_to_number(tf.reshape(idxs, [shape[0], -1] ), out_type=tf.int32) # None
        vals = kvs[:,1]
        vals = tf.string_to_number(tf.reshape(vals, [shape[0], -1] ), out_type=tf.float32)
        vals = tf.expand_dims(vals, 2)
        mask = tf.cast(tf.greater(idxs, 0), tf.int32)
        # tags embedding
        #TODO  "mean", "sqrtn" are not supported.
        embedding = tf.Variable(
            tf.random_normal([tag_max_size, tag_embedding_size], 0.0, 0.01), name="multi_tag_embedding")
        embedding = tf.nn.embedding_lookup(embedding, idxs)
        mask = tf.expand_dims(tf.cast(mask, tf.float32), 2)
        embedding = tf.multiply(embedding, mask)
        w_embedding = tf.multiply(embedding, vals)
        w_embedding = tf.reduce_sum(w_embedding, axis=1)
        # expand dims 
        w_embedding = tf.expand_dims(w_embedding, 1)

        bias = tf.Variable(
            tf.random_normal([1, 1], 0.0, 0.01), name="multi_tag_bais")
        bias = tf.reshape(tf.tile(tf.reshape(bias, [1]), [tf.shape(embedding)[0]]),[-1,1])
        bias = tf.expand_dims(bias, 1)

        return w_embedding, bias

    def _input_fn(self):
        COLS = [self.label_col_name] + self.features_config
        selected_cols = ','.join(COLS)
        if self.verbose > 0:
            print("select columns : %s" % selected_cols)

        def _expample_parser(*args):
            multi_tags_features = None
            if self.use_multi_tags:
                i = self.features_config.index(self.multi_tags_col_name[0]) + 1
                features = list(args[1:i])
                multi_tags_features = list(args[i:])
            else:
                features = list(args[1:])
            
            features = tf.string_join(features, self.kvs_delimiter)
            idxs, vals, mask = self._parse_sparse_feature(features)
            label = tf.string_to_number(args[0], out_type=tf.int32)
            
            if multi_tags_features:
                ret = [label,features, idxs, vals, mask] + multi_tags_features
            else:
                ret = [label,features, idxs, vals, mask]
            if self.mode == "predict":
                ret += args
            return ret

        # cross platform reader, support oss, odps and localhost
        reader = Reader(self.input_name, selected_cols, ",", self.task_index, self.worker_num)
        dataset = reader.read()
        dataset = dataset.map(map_func=_expample_parser)

        # dataset shuffule
        dataset = dataset.repeat(count=self.epoch)
        dataset = dataset.batch(self.batch_size)
        # gather_nd need
        self.iterator = dataset.make_initializable_iterator()
        v = self.iterator.get_next()
        # stack all feature
        self.feat_index = v[2]
        self.feat_value = v[3]
        self.mask = v[4]
        end_pos = 4
        self.label = tf.reshape(v[0], [-1, 1])
        if self.multi_tags_col_name:
            end_pos = 5 + len(self.multi_tags_col_name)
            self.multi_tags = v[5: end_pos]
        
        # prepare for predict operation
        if self.mode == "predict":
            rows = v[end_pos:]
            cols_name = [self.label_col_name] + self.features_config
            for i,name in enumerate(cols_name):
                self.feature_dict[name] = rows[i]

        # to survey positive label in batch data
        self.pos_num = tf.reduce_sum(self.label)

    def _parse_sparse_feature(self, features):
        # split kv string
        kvs = tf.string_split([features], self.kvs_delimiter)
        kvs = tf.string_split(kvs.values, self.kv_delimiter)
        key_values = tf.reshape(kvs.values, kvs.dense_shape)
        idxs, vals = tf.split(key_values, num_or_size_splits=2, axis=1)
        # kv values and mask, mask for some missing features
        idxs = tf.reshape(tf.string_to_number(idxs, out_type=tf.int32), [-1])
        vals = tf.reshape(tf.string_to_number(vals, out_type=tf.float32), [-1])
        mask = tf.cast(tf.greater(idxs, 0), tf.int32) 
        
        idx_pos = self.kv_map_t.lookup(tf.cast(idxs, dtype=tf.int64))
        idx_pos = tf.expand_dims(tf.cast(idx_pos, dtype=tf.int32),1)
        # move idxs to real model slot
        # input string may be like: 100:0.1,50:0.5,30:0.7
        # need move idxs to correct slot by kv_map setting
        # TODO unique repeat solt
        idxs = tf.scatter_nd(
                idx_pos, idxs, tf.constant([self.feature_size]))  # None * F
        vals = tf.scatter_nd(
                idx_pos, vals, tf.constant([self.feature_size]))
        mask = tf.reshape(tf.scatter_nd(
                idx_pos, mask,tf.constant([self.feature_size]))
            ,[1, self.feature_size]
            )
        idxs = tf.identity(idxs)
        vals = tf.identity(vals)
        mask = tf.identity(mask)
        
        return idxs, vals, mask

    def _build_graph(self, serving_mode):
        # random seed
        tf.set_random_seed(self.random_seed)

        # model config
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")

        # init variables
        self.weights = self._initialize_weights()
        
        # model
        self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"], self.feat_index)
        self.mask = tf.transpose(tf.cast(self.mask, tf.float32), [0,2,1])
        self.embeddings = tf.multiply(self.embeddings, self.mask) #None * F * K
        
        # weight to embedding
        feat_value = tf.reshape(self.feat_value, shape=[-1, self.feature_size, 1])
        self.embeddings = tf.multiply(self.embeddings, feat_value)
        
        # bias
        self.bias = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
        self.bias = tf.multiply(self.bias, self.mask)
        self.bias = tf.multiply(self.bias, feat_value)  # None * F * 1
        # use multi tags 
        if self.use_multi_tags:
            for multi_tags_feature_val in self.multi_tags:
                multi_tags_embeddings, multi_tags_bias = self._multi_tags_embedding(multi_tags_feature_val,
                        self.multi_tags_max_size, self.multi_tags_embedding_size)
                self.embeddings = tf.concat([self.embeddings, multi_tags_embeddings], axis=1)
                self.bias = tf.concat([self.bias, multi_tags_bias], axis=1)

        # first order term
        self.y_first_order = tf.reduce_sum(self.bias, 2)  # None * F
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F
        
        # second order term
        # sum_square part
        self.summed_features_emb = tf.reduce_sum(self.embeddings, 1)  # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # square_sum part
        self.squared_features_emb = tf.square(self.embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        # second order
        self.y_second_order = 0.5 * tf.subtract(
                self.summed_features_emb_square, self.squared_sum_features_emb
                )  # None * K
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

        # Deep component
        self.y_deep = tf.reshape(self.embeddings, 
                shape=[-1, self.all_feature_size * self.embedding_size]) # None * (F*K)
        self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
        for i in range(0, len(self.deep_layers)):
            self.y_deep = tf.matmul(self.y_deep, self.weights["layer_%d" %i]) # None * layer[i] * 1
            self.y_deep = tf.add(self.y_deep, self.weights["bias_%d"%i]) # None * layer[i] * 1
            if self.batch_norm:
                self.y_deep = self._batch_norm_layer(self.y_deep, 
                        train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer

        # DeepFM
        if self.use_fm and self.use_deep:
            concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)
        elif self.use_fm:
            concat_input = tf.concat([self.y_first_order, self.y_second_order], axis=1)
        elif self.use_deep:
            concat_input = self.y_deep

        self.out = tf.matmul(concat_input, self.weights["concat_projection"])
        self.out = tf.add(self.out, self.weights["concat_bias"])
        self.out = tf.nn.sigmoid(self.out)
        if serving_mode:
          return

        self.gs = tf.train.get_or_create_global_step()
        self.auc = tf.metrics.auc(labels=self.label, predictions=self.out)

        if self.mode == "train":
          # loss
          if self.loss_type == "logloss":
              self.loss = tf.losses.log_loss(self.label, self.out)
          elif self.loss_type == "mse":
              self.loss = tf.nn.l2_loss(tf.subtract(self.label, self.out))
          # l2 regularization on weights
          if self.l2_reg > 0:
              self.loss += tf.contrib.layers.l2_regularizer(
                  self.l2_reg)(self.weights["concat_projection"])
              if self.use_deep:
                  for i in range(len(self.deep_layers)):
                      self.loss += tf.contrib.layers.l2_regularizer(
                          self.l2_reg)(self.weights["layer_%d"%i])

          # optimizer, default is adam
          if self.optimizer_type == "adam":
              self.optimizer = tf.train.AdamOptimizer(
                      learning_rate=self.learning_rate,
                      beta1 = self.adam_beta1, beta2=self.adam_beta2,
                      epsilon=self.adam_epsilon)
          elif self.optimizer_type == "adagrad":
              self.optimizer = tf.train.AdagradOptimizer(
                      learning_rate=self.learning_rate,
                      initial_accumulator_value=self.adagrad_initial_accumulator_value)
          elif self.optimizer_type == "gd":
              self.optimizer = tf.train.GradientDescentOptimizer(
                      learning_rate=self.learning_rate)
          elif self.optimizer_type == "momentum":
              self.optimizer = tf.train.MomentumOptimizer(
                      learning_rate=self.learning_rate, momentum=self.momentum)
          
          # defalt is async, this options is usefull for 
          if self.sync_type == "sync":
              self.optimizer = tf.train.SyncReplicasOptimizer(
                      self.optimizer,
                      replicas_to_aggregate=self.cluster.num_tasks("worker"),
                      total_num_replicas=self.cluster.num_tasks("worker"))
              self.sync_opt_hook = self.optimizer.make_sesion_run_hook(
                                        (self.task_index == 0),num_tokens=0)

          self.optimizer = self.optimizer.minimize(self.loss, global_step=self.gs)
        elif self.mode == "predict":
            self.v_list = []
            self.v_list.append(tf.reshape(self.out, [-1]))
            for (feature_name, v) in self.feature_dict.items():
                self.v_list.append(v)
            # create table writer, just for odps table on pai platform
            if hasattr(tf, "TableRecordWriter") and self.output_name.startswith("odps"):
                writer = tf.TableRecordWriter(self.output_name, slice_id=self.task_index)
                self.write_to_table = writer.write(range(self.all_feature_size + 1), self.v_list)
                #self.close_table = writer.close()
            return
        
        self.avg_loss = tf.div(self.loss, self.batch_size)
        # init tf.Saver
        self.saver = tf.train.Saver(sharded=True)
        # summary infomation
        self.summary_merged, self.summary_writer = self._add_summary(
                self.checkpoint_dir, tf.get_default_graph())
        
        # number of params
        total_parameters = 0
        for variable in self.weights.values():
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        if self.verbose > 0:
            print("model params number: %d" % total_parameters)
            print("model total size: %d" % (total_parameters * 4))


    def _init_session(self):
        # create Session
        config = self.get_tf_run_config()
        config.gpu_options.allow_growth = True
        stop_hook = tf.train.StopAtStepHook(num_steps=self.num_steps)
        # dataset initializer with hook
        class InitHook(SessionRunHook):
            def __init__(self, initializer):
                self.initializer = initializer
            def after_create_session(self, sess, coord):
                sess.run(self.initializer.initializer)

        init_table_and_input_hook = InitHook(self.iterator)

        hooks = [stop_hook, init_table_and_input_hook]
        if self.sync_opt_hook:
            hooks.append(self.sync_opt_hook)
        # save/restore to/from checkpoint_dir
        sess = tf.train.MonitoredTrainingSession(
                master=self.get_master_target(), 
                is_chief=(self.task_index <= 0),
                checkpoint_dir=self.checkpoint_dir,
                hooks=hooks, config=config)
        return sess

    def _initialize_weights(self):
        weights = dict()

        # embeddings
        weights["feature_embeddings"] = tf.Variable(
            tf.random_normal([self.feature_max_size, self.embedding_size], 0.0, 0.01),
            name="feature_embeddings")  # feature_max_size * K
        weights["feature_bias"] = tf.Variable(
            tf.random_uniform([self.feature_max_size, 1], 0.0, 1.0), name="feature_bias")  # feature_max_size * 1

        # deep layers
        num_layer = len(self.deep_layers)
        input_size = self.all_feature_size * self.embedding_size
        glorot = np.sqrt(2.0 / (input_size + self.deep_layers[0]))
        weights["layer_0"] = tf.Variable(
            np.random.normal(loc=0, scale=glorot, size=(input_size, self.deep_layers[0])), dtype=np.float32)
        weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[0])),
                                                        dtype=np.float32)  # 1 * layers[0]
        for i in range(1, num_layer):
            glorot = np.sqrt(2.0 / (self.deep_layers[i-1] + self.deep_layers[i]))
            weights["layer_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(self.deep_layers[i-1], self.deep_layers[i])),
                dtype=np.float32)  # layers[i-1] * layers[i]
            weights["bias_%d" % i] = tf.Variable(
                np.random.normal(loc=0, scale=glorot, size=(1, self.deep_layers[i])),
                dtype=np.float32)  # 1 * layer[i]

        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.all_feature_size + self.embedding_size + self.deep_layers[-1]
        elif self.use_fm:
            input_size = self.all_feature_size + self.embedding_size
        elif self.use_deep:
            input_size = self.deep_layers[-1]
        glorot = np.sqrt(2.0 / (input_size + 1))
        weights["concat_projection"] = tf.Variable(
                        np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
                        dtype=np.float32)  # layers[i-1]*layers[i]
        weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32)

        return weights


    def _batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def _add_summary(self, path, graph):
        tf.summary.scalar("loss", self.loss)
        tf.summary.scalar("avg loss", self.avg_loss)
        tf.summary.scalar("global/auc", self.auc[0])
        #TODO gradient summary
        summary_writer = tf.summary.FileWriter(path, graph)
        summary_merged = tf.summary.merge_all()
        return summary_merged,summary_writer

    def train_and_evaluate(self):
        with self.graph.as_default(), self._init_session() as sess:
            pos_num_total=0
            global_step = -1;
            while not sess.should_stop():
                if global_step % 500 == 0 and self.task_index == 0:
                    eval_feed_dict = {self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                           self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                           self.train_phase: False}
                    run = [self.loss, self.avg_loss, self.auc, self.gs, self.pos_num, self.summary_merged]
                    loss, avg_loss, auc, gs, pos_num, summary_str = sess.run(
                          run,
                          feed_dict=eval_feed_dict)
                    logging.info("global steps: %s, loss: %s, avg loss: %s, auc: %s, positive sample num: %s", 
                        gs, loss, avg_loss, auc[0], pos_num)
                    self.summary_writer.add_summary(summary_str, gs)
                else:
                    feed_dict = {self.dropout_keep_fm: self.dropout_fm,
                               self.dropout_keep_deep: self.dropout_deep,
                               self.train_phase: True}
                    loss, avg_loss, opt, gs, pos_num = sess.run(
                          [self.loss, self.avg_loss, self.optimizer, self.gs, self.pos_num],
                          feed_dict=feed_dict)
                    logging.info("global steps: %s, loss: %s, positive sample num: %s", gs, loss, pos_num)
                global_step = gs
                if pos_num > 0:
                    pos_num_total = pos_num_total + pos_num
        logging.info("end of train/evaluate")
        if self.task_index == 0:
            self.export_saved_model()

    def predict(self):
        feed_dict = {self.dropout_keep_fm: [1.0] * len(self.dropout_fm),
                         self.dropout_keep_deep: [1.0] * len(self.dropout_deep),
                         self.train_phase: False}
        # platform adaptation
        if hasattr(tf, "TableRecordWriter") and self.output_name.startswith("odps"):
            def _write(sess):
                sess.run([self.write_to_table], feed_dict=feed_dict)
        else:
            writer = Writer(self.output_name, self.worker_num, self.task_index)
            def _write(sess):
                rows = sess.run(self.v_list, feed_dict=feed_dict)
                rows = [np.expand_dims(r, 1) for r in rows]
                rows = np.concatenate(rows, axis=1)
                rows = rows.astype(np.str)
                rows = rows.tolist()
                for lines in rows:
                    writer.write(lines)

        with self.graph.as_default(), self._init_session() as sess:
            step = 0
            while not sess.should_stop():
                step += 1
                _write(sess)
                if step%1000 == 0:
                    logging.info('predict %d steps done, process records %d',
                            step, step * self.batch_size)
        logging.info("end of predict")

    def _build_signature_def(self):
        inputs = dict()
        outputs = dict()

        features = tf.placeholder(dtype=tf.string, shape=[None], name="features")
        self._build_kv_map()
        def map_parse(f):
            v = self._parse_sparse_feature(f)
            return v
        v = tf.map_fn(map_parse, features, dtype=(tf.int32,tf.float32,tf.int32))
        # stack all feature
        self.feat_index = v[0]
        self.feat_value = v[1]
        self.mask = v[2]
        # merge normal features, multi tags feature is independent
        inputs["features"] = tf.saved_model.utils.build_tensor_info(features)
        
        if self.use_multi_tags:
            multi_tags_feature = []
            for feature_name in self.multi_tags_col_name:
                tag_feature = tf.placeholder(dtype=tf.string, shape=[None], name=feature_name)
                inputs[feature_name] = tf.saved_model.utils.build_tensor_info(tag_feature) 
                multi_tags_feature.append(tag_feature)
            self.multi_tags = multi_tags_feature

        self._build_graph(serving_mode=True)
        outputs["score"] = tf.saved_model.utils.build_tensor_info(self.out)
        
        def_utils = tf.saved_model.signature_def_utils
        signature_def = def_utils.build_signature_def(inputs, outputs, "signature")
        return signature_def

    def export_saved_model(self):
        if self.task_index == 0:
          with tf.Graph().as_default() as g:
            signature_def = self._build_signature_def()
            with tf.Session() as sess:
              self.saver = tf.train.Saver(sharded=True)
              checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_dir)
              self.saver.restore(sess, checkpoint_path)
              builder = tf.saved_model.builder.SavedModelBuilder(
                  os.path.join(self.checkpoint_dir, 'saved_model',
                               datetime.datetime.now().strftime('%Y%m%d-%H%M%S')))
              builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={"deepfm_predict": signature_def},
                assets_collection=tf.get_collection(
                      tf.GraphKeys.ASSET_FILEPATHS),
                )
              builder.save()
              logging.info("end of export_saved_model")
