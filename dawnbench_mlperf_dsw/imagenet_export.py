
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import oss2
import argparse
import logging
import tensorflow as tf  # pylint: disable=g-bad-import-order
from imagenet_main import *
from utils.export import export
from quantize.quant_hooks_v2 import QuantHook

import tensorflow_io.oss
import tensorflow_io.oss.python.ops.ossfs_ops


parser = argparse.ArgumentParser(description='TF Graph Test')
parser.add_argument('--model-dir', default='./ft_model/', type=str,
                    help='model path')
parser.add_argument('--data-dir', default='/data/ImageNet_TFRecorder', type=str,
                    help='model path')
parser.add_argument('--pickle-model', default='./gap_save/gap_pruned.pkl', type=str,
                    help='pickle path')
parser.add_argument('--final-size', default=2048, type=int, metavar='N',
                    help='final size')
parser.add_argument('--export-dir', default='./export/', type=str,
                    help='export path')
parser.add_argument('--enable-quantize', '-eqz', action='store_true',
                    help='if True quantization is enabled.')
parser.add_argument('--q-bits', default=4, type=int, metavar='N',
                    help='quantization bits')
parser.add_argument("--oss_load", "-osl", action='store_true',
                    help="[default: %(default)s] oss_load: If True dataset is loaded from oss.")
parser.add_argument('--log-name', type=str, default='export_eval', help='log name')
args = parser.parse_args()

# set logging system
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='export/'+args.log_name+'.log')
logging.info(args)



_BATCH_SIZE = 256
_LABEL_CLASSES = 1001
_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}
_NUM_TRAIN_FILES = 8
_SHUFFLE_BUFFER = 1500

_ACCESS_ID = "<TBD>"
_ACCESS_KEY = "<TBD>"
_HOST = "<TBD>"
_BUCKET = "<TBD>"

if args.oss_load:
  auth = oss2.Auth(_ACCESS_ID, _ACCESS_KEY)
  bucket = oss2.Bucket(auth, _HOST, _BUCKET)

shape=[_DEFAULT_IMAGE_SIZE, _DEFAULT_IMAGE_SIZE, _NUM_CHANNELS]

# create estimator
session_config = tf.ConfigProto(
    inter_op_parallelism_threads=0,
    intra_op_parallelism_threads=0,
    allow_soft_placement=True)

distribution = tf.contrib.distribute.OneDeviceStrategy('device:GPU:0')
run_config = tf.estimator.RunConfig(train_distribute=distribution,
                                    session_config=session_config)

classifier = tf.estimator.Estimator(
    model_fn=imagenet_model_fn,
    model_dir=args.model_dir, config=run_config,
    params={
          'resnet_size': 26,
          'final_size': args.final_size,
          'pickle_model': args.pickle_model,
          'random_init': False,
          'data_format': "channels_last",
          'batch_size': _BATCH_SIZE,
          'train_epochs': 1,
          'version': 34,
          'version_t': 1,
          'loss_scale': 1,
          'gap_train': False,
          'gap_lambda': 0.00001,
          'gap_ft': False,
          'gap_start': 0,
          'dtype': tf.float32,
          'learn_rate': 0.1,
          'label_smoothing': 0.,
          'enable_lars': False,
          'enable_cos': False,
          'cos_alpha': 0.001,
          'warm_up': False,
          'weight_decay': 0.00001,
          'fine_tune': False,
          'enable_kd': False,
          'kd_size': 50,
          'temp_dst': 1.,
          'w_dst': 1.,
          'mix_up': False,
          'mx_mode': 0,
          'enable_quantize': False,
          'online_quantize': False,
          'enable_at': False,
          'w_at': 2.,
      })

# create input_fn
def input_fn_eval():
  return input_fn(
      is_training=False,
      data_dir=args.data_dir,
      batch_size=_BATCH_SIZE,
      num_epochs=1,
      dtype=tf.float32,
      oss_load=args.oss_load
  )

eval_hooks = None
if args.enable_quantize:
    quant_eval_hook = QuantHook(bits=args.q_bits)
    eval_hooks = [quant_eval_hook]

# evaluation
eval_results = classifier.evaluate(input_fn=input_fn_eval,
                                   steps=None,
                                   hooks=eval_hooks)
logging.info(eval_results)

# save the model
logging.info('Export the saved model!')
input_receiver_fn = export.build_tensor_serving_input_receiver_fn(
          shape, batch_size=None, dtype=tf.float32)

classifier.export_saved_model(args.export_dir, input_receiver_fn)
logging.info('Finished export!')
