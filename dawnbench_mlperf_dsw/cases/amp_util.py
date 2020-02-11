import sys
import pdb
import time
import tensorflow as tf
import numpy as np
from tensorflow.core.protobuf import rewriter_config_pb2

def load_graph(pb_file_path):
    pb_graph_def = tf.GraphDef()
    with open(pb_file_path, 'rb') as f:
        pb_graph_def.ParseFromString(f.read())
    return pb_graph_def


def save_graph(graph_def, pb_file_path):
  with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    f.write(graph_def.SerializeToString())
  return pb_file_path


def graph_to_pb(sess, graph, output_names):
    input_graph_def = graph.as_graph_def()
    from tensorflow.python.framework import graph_util
    # We use a built-in TF helper to export variables to constant
    output_graph_def = graph_util.convert_variables_to_constants(
                        sess,
                        input_graph_def,
                        output_node_names=output_names)
    return output_graph_def


def load_savedmodel(model_path):
    from tensorflow.python.saved_model import loader_impl
    saved_model = loader_impl._parse_saved_model(model_path)
    tf.reset_default_graph()
    cfg = tf.ConfigProto(allow_soft_placement=True,
                         log_device_placement=False)
    cfg.gpu_options.allow_growth = True
    with tf.Session(config=cfg) as sess:
        tags = saved_model.meta_graphs[0].meta_info_def.tags
        meta_graph_def = tf.saved_model.loader.load(sess, tags, model_path)
        sdef_key = meta_graph_def.signature_def.keys()[0]
        tmp_outputs = meta_graph_def.signature_def[sdef_key].outputs.values()
        model_outputs = [v.name[:-2] for v in tmp_outputs]
        graph_def = tf.graph_util.convert_variables_to_constants(
                        sess, sess.graph_def, model_outputs)
    graph_def = tf.graph_util.extract_sub_graph(graph_def, model_outputs)
    for i,node in enumerate(graph_def.node):
        if '_class' in node.attr.keys():
            node.attr.pop('_class')
    return graph_def


def getSessConfig():
    gpu_options = tf.GPUOptions(allow_growth=True,
                                allocator_type='BFC',
                                per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options=gpu_options)
    return config

def getOffLayoutConfig():
    rewrite_options = rewriter_config_pb2.RewriterConfig(layout_optimizer=rewriter_config_pb2.RewriterConfig.OFF)
    graph_options = tf.GraphOptions(rewrite_options=rewrite_options)
    gpu_options = tf.GPUOptions(allow_growth=True,
                                allocator_type='BFC',
                                per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(log_device_placement=False,
                            allow_soft_placement=True,
                            gpu_options=gpu_options,
                            graph_options = graph_options)
    return config

def print_graph_infor(graph_def, logging):
    for i,n in enumerate(graph_def.node):
        logging.info(n.name)
        #if (n.op == 'Conv2D'):
        #    logging.info("Name of the node - %s" % n.name)
        #    for j,n_input in enumerate(n.input):
        #        logging.info("input[{}] of the node {} - {}".format(j, n.name, n_input))


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


def extract_graph_node(graph_def, logging):
    activate_dict = {}
    kernel_dict = {}
    act_kernel_list = []
    op_kernel_list = []
    for i,n in enumerate(graph_def.node):
        if (n.op in ['Conv2D', 'MatMul']): # 'MatMul'
            logging.info("Name of the node - %s" % n.name)
            act_name = n.input[0]+':0'
            activate_dict[act_name] = [] # input tensor name, scaling factor
            logging.info("input[{}] of the node {} - {}".format(0, n.name, act_name))

            if 'read' in n.input[1]:
                kernel_name = n.input[1][:-5] # kernel const node name
            else:
                kernel_name = n.input[1] # kernel const node name
            logging.info("input[{}] of the node {} - {}".format(1, n.name, kernel_name))
            kernel_dict[kernel_name] = 1. # kernel const node name, scaling factor
            act_kernel_list.append((act_name, kernel_name))
            op_kernel_list.append((act_name, n.name, kernel_name))

    return activate_dict, kernel_dict, act_kernel_list, op_kernel_list
