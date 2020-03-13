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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pprint

import tensorflow as tf
from DeepFM import DeepFM

def python_style_str_param_transform(p):
    print("python style transform,",p)
    try:
        x = eval(p)
        return x
    except:
        raise Exception("python str param format error, %s" %(p))

def optimizer_param_check(p):
    if p not in ("adam", "adagrad", "gd", "monmentum"):
        raise Exception("optimizer param error")
    return p

def feature_cols_param_check(p):
    cols = p.split(",")
    if len(cols) <= 0:
        raise Exception("feature cols param error")
    return cols

def ModelStatistics():
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    for v in tf.trainable_variables():
        print(v.name, v.device, size(v))
    print("total model size:", sum(size(v) for v in tf.trainable_variables()))


tf.app.flags.DEFINE_string("tables", "", "tables info")
tf.app.flags.DEFINE_string("feature_cols", "", "feautre columns")
tf.app.flags.DEFINE_string("multi_tags_cols", None, "multi tags columns")
tf.app.flags.DEFINE_string("kv_map", "", "kv map for kv features transform to feature fild")
tf.app.flags.DEFINE_integer("feature_max_size", -1, "max key size of feature")
tf.app.flags.DEFINE_integer("multi_tags_max_size", 100, "multi tags max key size of feature")
tf.app.flags.DEFINE_string("label_col_name", "", "label column name")
tf.app.flags.DEFINE_string("kvs_delimiter", ",", "delimiter between to kvs") 
tf.app.flags.DEFINE_string("kv_delimiter", ":", "delimiter between key and value")
tf.app.flags.DEFINE_integer("task_index", 0, "Worker task index")
tf.app.flags.DEFINE_string("ps_hosts", None, "ps hosts")
tf.app.flags.DEFINE_string("worker_hosts", None, "worker hosts")
tf.app.flags.DEFINE_string("job_name", 'worker', "job name: worker or ps")
tf.app.flags.DEFINE_string("checkpointDir", '', "checkpoint dir")
tf.app.flags.DEFINE_string("outputs", '', "output table")
tf.app.flags.DEFINE_string("mode", 'train', "train or predict")
tf.app.flags.DEFINE_boolean("use_fm", True, 'use fm or not')
tf.app.flags.DEFINE_boolean("use_deep", True, 'use deep or not')
tf.app.flags.DEFINE_string("sync_type", "async", 'sync or not')
tf.app.flags.DEFINE_integer("embedding_size", 8, "ebmedding size")
tf.app.flags.DEFINE_integer("num_steps", 100*1000*1000, "max num steps")
tf.app.flags.DEFINE_string("dropout_fm", "[1.0, 1.0]", "dropout param in parts of fm, python list style, e.g: [0.5,0.5]")
tf.app.flags.DEFINE_string("deep_layers", "[32, 32]", "deep layers")
tf.app.flags.DEFINE_string("dropout_deep", "[0.5, 0.5, 0.5]","dropout param in parts of fm, python list style, e.g: [0.5,0.5]")
tf.app.flags.DEFINE_integer("batch_size", 128, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "laerning rate")
tf.app.flags.DEFINE_string("optimizer_type", "adam", "type of optimizer")
tf.app.flags.DEFINE_boolean("batch_norm", True, "batch norm or not")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.995,"")
tf.app.flags.DEFINE_float("l2_reg", 0.01, "L2 regularization")
tf.app.flags.DEFINE_integer("epoch", 1, "number of epoch")
tf.app.flags.DEFINE_float("adam_beta1", 0.9, "adam optimizer beta1 param") 
tf.app.flags.DEFINE_float("adam_beta2", 0.999, "adam optimizer beta2 param")
tf.app.flags.DEFINE_float("adam_epsilon", 1e-8,"adam optimizer")
tf.app.flags.DEFINE_float("adagrad_initial_accumulator_value", 1e-8, "adagrad optimizer initial accumulator param")
tf.app.flags.DEFINE_float("momentum", 0.95, "momentum optimizer param")
tf.app.flags.DEFINE_boolean("log_verbose", True, "print verbose log or not")
tf.app.flags.DEFINE_integer("random_seed", 123456, "random seed")

FLAGS = tf.app.flags.FLAGS

def distribute_params():
    # distribute params process
    worker_num = 1
    cluster = None
    server = None

    ps_hosts = FLAGS.ps_hosts
    worker_hosts = FLAGS.worker_hosts
    if ps_hosts or worker_hosts:
        ps_hosts = ps_hosts.split(",")
        worker_hosts = worker_hosts.split(",")
        worker_num = len(worker_hosts)

        cluster = tf.train.ClusterSpec(
                {"ps": ps_hosts, "worker": worker_hosts}
                )
        server = tf.train.Server(cluster,
                job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    return worker_num, cluster, server

def main(argv):
    # sartup with different role
    worker_num, cluster, server = distribute_params()
    if FLAGS.job_name == "ps":
      server.join()
    elif len(FLAGS.job_name) <=0 or FLAGS.job_name == "worker":
        dfm_params = {
            "use_fm": FLAGS.use_fm,
            "use_deep": FLAGS.use_deep,
            "embedding_size": FLAGS.embedding_size,
            "dropout_fm": python_style_str_param_transform(FLAGS.dropout_fm),
            "deep_layers": python_style_str_param_transform(FLAGS.deep_layers),
            "dropout_deep": python_style_str_param_transform(FLAGS.dropout_deep),
            "deep_layers_activation": tf.nn.relu,
            "batch_size": FLAGS.batch_size,
            "learning_rate": FLAGS.learning_rate,
            "optimizer_type": optimizer_param_check(FLAGS.optimizer_type),
            "batch_norm": FLAGS.batch_norm,
            "batch_norm_decay": FLAGS.batch_norm_decay,
            "l2_reg": FLAGS.l2_reg,
            "epoch": FLAGS.epoch,
            "num_steps": FLAGS.num_steps,
            "random_seed": FLAGS.random_seed,
            "server": server,
            "cluster": cluster,
            "task_index": FLAGS.task_index,
            "worker_num": worker_num,
            "input_name": FLAGS.tables,
            "feature_cols": feature_cols_param_check(FLAGS.feature_cols),
            "kv_map":python_style_str_param_transform(FLAGS.kv_map),
            "feature_max_size": FLAGS.feature_max_size,
            "label_col_name": FLAGS.label_col_name,
            "kvs_delimiter": FLAGS.kvs_delimiter,
            "kv_delimiter": FLAGS.kv_delimiter,
            "sync_type": FLAGS.sync_type,
            "checkpoint_dir": FLAGS.checkpointDir,
            "output_name": FLAGS.outputs,
            "mode": FLAGS.mode,
            "multi_tags_col_name" : FLAGS.multi_tags_cols,
            "multi_tags_max_size": FLAGS.multi_tags_max_size,
            "adam_beta1" : FLAGS.adam_beta1,
            "adam_beta2" : FLAGS.adam_beta2,
            "adam_epsilon" : FLAGS.adam_epsilon,
            "adagrad_initial_accumulator_value" : FLAGS.adagrad_initial_accumulator_value,
            "momentum" : FLAGS.momentum
        }
        if FLAGS.log_verbose:
            tf.logging.set_verbosity(tf.logging.INFO)

        pprint.pprint(dfm_params)
        dfm = DeepFM(**dfm_params)
        ModelStatistics()
        if FLAGS.mode == 'train':
          dfm.train_and_evaluate()
        elif FLAGS.mode == 'predict':
          dfm.predict()
        else:
          print("mode must be train or predcit. please add -DuserDefinedParameters='--mode=train/predict'")

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("show DeepFM help info below:")
        print(FLAGS.__dict__['__wrapped'])
        print("\n\nplease set run param to run DeepFM!")
        exit(0)
    tf.app.run(main)
