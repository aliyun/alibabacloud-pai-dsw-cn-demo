
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import logging
import tensorflow as tf  # pylint: disable=g-bad-import-order

from gap_tools import gap_train_prune

parser = argparse.ArgumentParser(description='TF GAP Test')
parser.add_argument('--model', default='./model/model.ckpt', type=str,
                    help='model path')
parser.add_argument('--save_path', default='./gap_save', type=str,
                    help='gap save path')
parser.add_argument('--slim_ratio', default=0.5, type=float,
                    help='gap slim ratio (default: 0.5)')
parser.add_argument('--mask-len', '-ml', dest='ml', action='store_true', help='mask len')
parser.add_argument('--full-save', '-fs', dest='fs', action='store_true', help='full save')
parser.add_argument('--ver', default=1, type=int, help='resnet version')
parser.add_argument('--var-scope', type=str, default='resnet_model', help='var scope')
parser.add_argument('--log-name', type=str, default='gap_prune', help='log name')
args = parser.parse_args()

# set logging system
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='log/'+args.log_name+'.log')
logging.info(args)


if __name__ == '__main__':
    logging.info('GAP pruning starts!')
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_file = os.path.join(args.save_path, 'gap_pruned.pkl')
    full_file = os.path.join(args.save_path, 'gap_unpruned.pkl')
    data_dict = gap_train_prune.gap_perform_pruning(args.model, save_file, 'gap',
                                            args.slim_ratio, args.ml, args.fs, full_file,
                                            args.var_scope, args.ver)
    logging.info('GAP pruning ends!')
