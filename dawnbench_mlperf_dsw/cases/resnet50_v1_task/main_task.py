import os
import sys
import pdb
import time
import collections
import tensorflow as tf
import numpy as np

root_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(root_dir + '/../')
from amp_util import flops_stat, load_graph, getSessConfig, getOffLayoutConfig

class Resnet50V1Task(object):
    def __init__(self, height=224, width=224):
        self.logging = None
        self.name = 'resnet50_v1'

        #config quantization
        self.height = height
        self.width = width
        self.calib_file = os.path.join(root_dir, '../../calib/calib_{}_64.npy'.format(self.height))

    def calib_dump(self, infer_graph_def, tf_layopt=True, act_dict=None):
        with tf.Graph().as_default() as graph:
            # import graph_def to tf.Graph
            _ = tf.import_graph_def(infer_graph_def, name='')
            flops_stat(graph, self.logging)

            if tf_layopt is True:
                config = getSessConfig()
            else:
                config = getOffLayoutConfig()

            with tf.Session(config=config) as sess:
                # dump calibration data for activations
                act_list = list()
                for act_name in act_dict:
                    act_list.append(act_name)

                calib_data = np.load(self.calib_file, allow_pickle=True)

                for i, calib_dict in enumerate(calib_data):
                    acts = sess.run(act_list, feed_dict=calib_dict)
                    for act_name, act in zip(act_list, acts):
                        act_dict[act_name].append(act)

                return act_dict


if __name__ == '__main__':

    tmp_task = MobilenetV1Task()
    graph_def = load_graph(tmp_task.model_file)
    tmp_task.evaluate(graph_def)
