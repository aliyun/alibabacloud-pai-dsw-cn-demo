
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf  # pylint: disable=g-bad-import-order
from tensorflow.python.training import saver
from tensorflow.python.framework import ops

from tensorflow.contrib.nccl.python.ops import nccl_ops
nccl_ops._maybe_load_nccl_ops_so()

parser = argparse.ArgumentParser(description='TF Graph Test')
parser.add_argument('--model', default='./model/model.ckpt', type=str,
                    help='model path')
parser.add_argument('--log-name', type=str, default='graph_stat', help='log name')
args = parser.parse_args()

# set logging system
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/'+args.log_name+'.log')
logging.info(args)


def flops_stat(graph, logging):
    flops_ori = 0
    params_ori = 0
    for op in graph.get_operations():
        if (op.type in ['Conv2D', 'MatMul']) and op.name.startswith('resnet_model'):
            # NHWC, HWMN
            if op.type == 'Conv2D':
                flops_layer= op.outputs[0].shape[1] * op.outputs[0].shape[2] * \
                             np.prod(op.inputs[1].shape)
            else:
                flops_layer= np.prod(op.inputs[1].shape)
            flops_layer *= 2
            params_layer = np.prod(op.inputs[1].shape)
            flops_ori += flops_layer
            params_ori += params_layer
            logging.info('Flops: {:}, Params: {:}, Input Shape: {:}, Output Shape: {:}, Kernel Shape: {:} of layer: {:}'.
                         format(flops_layer, params_layer, op.inputs[0].shape, op.outputs[0].shape, op.inputs[1].shape, op.name))

        elif op.type in ['MaxPool']:
            pool_size = 3
            flops_layer= op.outputs[0].shape[1] * op.outputs[0].shape[2] * \
                         op.outputs[0].shape[3] * pool_size * pool_size
            flops_ori += flops_layer
            logging.info('Flops: {:} of layer: {:}'.format(flops_layer, op.name))
    logging.info('Total flops: {:}, and total params: {:}'.format(flops_ori, params_ori))


if __name__ == '__main__':
    logging.info('Graph stat starts!')
    meta_graph = saver.import_meta_graph(args.model+'.meta', clear_devices=True)
    graph = ops.get_default_graph()

    for op in graph.get_operations():
        if (op.type == 'FusedBatchNorm') and (not 'tower' in op.name):
            logging.info('Node: {}, shape: {}'.format(op.name, op.outputs[0].shape.as_list()))

    flops_stat(graph, logging)
    logging.info('Graph stat ends!')
