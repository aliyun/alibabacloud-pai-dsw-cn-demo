import sys
import os
import time
import pdb
import argparse
import logging
import tensorflow as tf
import numpy as np

from collections import OrderedDict
from task_util import create_task
from quant_util import *


parser = argparse.ArgumentParser(description='TF ImageNet Test')
parser.add_argument('--model', default='/app/export/', type=str,
                    help='model path')
parser.add_argument('--load', default=1, type=int,
                    help='load method, 0-pb, 1-savedmodel')
parser.add_argument('--task', default='imagenet', type=str,
                    help='evaluation task type')
parser.add_argument('--qmode', default=0, type=int,
                    help='quantization mode')
parser.add_argument('--method', default='KL', type=str,
                    help='offline quantize mode for activation')
parser.add_argument('--size', default=224, type=int,
                    help='input image size (default: 224)')
parser.add_argument('--bits', default=8, type=int,
                    help='quantization bits (default: 8)')
parser.add_argument('--log-name', type=str, default='test', help='log name')
args = parser.parse_args()

# set logging system
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename='./log/'+args.log_name+'.log')
logging.info(args)


if __name__ == '__main__':

  task = create_task(args.task, args.size, args.size)
  task.logging = logging

  if args.load==0:
      graph_def = load_graph(args.model)
  elif args.load==1:
      graph_def = load_savedmodel(args.model)

  print_graph_infor(graph_def, logging)
  kl_calib(graph_def, task, args, logging=logging)
