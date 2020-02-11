"""
  test code for GAP tools
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.training import moving_averages
from tensorflow.python.platform import test

from tensorflow.contrib.model_distillation.gap_tools import gap_train_prune
from tensorflow.contrib.model_distillation.gap_tools import gap_finetune

class GAPTest(test.TestCase):
  """ GAP Test Class. """
  def setUp(self):
    self._gen_vars()

  def _gen_vars(self):
    """ Function to generate all the variable values in numpy format. """
    # variables value generate
    self.conv1_kernel, self.conv1_bias, \
        self.bn1_gamma, self.bn1_beta, \
        self.bn1_moving_mean, self.bn1_moving_variance = \
            self._gen_numpy_var(3, 3, 16)
    self.conv2a_1_kernel, self.conv2a_1_bias, \
        self.bn2a_1_gamma, self.bn2a_1_beta, \
        self.bn2a_1_moving_mean, self.bn2a_1_moving_variance = \
            self._gen_numpy_var(1, 16, 16)
    self.conv2a_2_kernel = list()
    self.conv2a_2_bias = list()
    for card in range(4):
      kernel, bias, _, _, _, _ = self._gen_numpy_var(3, 4, 4)
      self.conv2a_2_kernel.append(kernel)
      self.conv2a_2_bias.append(bias)
    _, _, self.bn2a_2_gamma, self.bn2a_2_beta, \
        self.bn2a_2_moving_mean, self.bn2a_2_moving_variance = \
            self._gen_numpy_var(3, 16, 16)
    self.conv2a_3_kernel, self.conv2a_3_bias, \
        self.bn2a_3_gamma, self.bn2a_3_beta, \
        self.bn2a_3_moving_mean, self.bn2a_3_moving_variance = \
            self._gen_numpy_var(1, 16, 64)
    self.conv2b_kernel, self.conv2b_bias, \
        self.bn2b_gamma, self.bn2b_beta, \
        self.bn2b_moving_mean, self.bn2b_moving_variance = \
            self._gen_numpy_var(1, 16, 64)

    self.conv3_kernel, self.conv3_bias, \
        self.bn3_gamma, self.bn3_beta, \
        self.bn3_moving_mean, self.bn3_moving_variance = \
            self._gen_numpy_var(1, 64, 256)


  def _gen_numpy_var(self, kernel_size, input_size, output_size):
    kernel = np.random.rand(kernel_size, kernel_size, input_size, output_size)
    bias = np.random.rand(output_size)
    gamma = np.random.rand(output_size)
    beta = np.random.rand(output_size)
    moving_mean = np.random.rand(output_size)
    moving_variance = np.random.rand(output_size)
    return kernel, bias, gamma, beta, moving_mean, moving_variance

  def _conv2d(self, input_x, scope, kernel_value, bias_value):
    """ Wrapper function for convolution. """
    with variable_scope.variable_scope(scope):
      kernel_initial = init_ops.constant_initializer(\
                kernel_value,
                dtypes.float32)
      kernel = gap_finetune.get_variable(\
                name='kernel',
                shape=kernel_value.shape,
                dtype=dtypes.float32,
                initializer=kernel_initial,
                gap=self.gap,
                gap_vars=self.gap_vars)
      bias_initial = init_ops.constant_initializer(bias_value, dtypes.float32)
      bias = gap_finetune.get_variable(\
                name='bias',
                shape=bias_value.shape,
                dtype=dtypes.float32,
                initializer=bias_initial,
                gap=self.gap,
                gap_vars=self.gap_vars)
      out = nn_ops.conv2d(input_x, kernel, strides=[1, 1, 1, 1], padding='SAME')
      out = nn_ops.bias_add(out, bias)
      return out

  def _batchnorm(self, input_x, scope, \
                 gamma_value, beta_value,\
                 moving_mean_value, moving_variance_value,\
                 is_training):
    """ Wrapper function for batch normalization. """
    with variable_scope.variable_scope(scope):
      gamma_initial = init_ops.constant_initializer(gamma_value, dtypes.float32)
      gamma = gap_finetune.get_variable(\
                name='gamma',
                shape=gamma_value.shape,
                dtype=dtypes.float32,
                initializer=gamma_initial,
                gap=self.gap,
                gap_vars=self.gap_vars)
      beta_initial = init_ops.constant_initializer(beta_value, dtypes.float32)
      beta = gap_finetune.get_variable(\
                name='beta',
                shape=beta_value.shape,
                dtype=dtypes.float32,
                initializer=beta_initial,
                gap=self.gap,
                gap_vars=self.gap_vars)
      moving_mean_initial = init_ops.constant_initializer(\
                moving_mean_value,
                dtypes.float32)
      moving_mean = gap_finetune.get_variable(\
                name='moving_mean',
                shape=moving_mean_value.shape,
                dtype=dtypes.float32,
                initializer=moving_mean_initial,
                gap=self.gap,
                gap_vars=self.gap_vars)
      moving_variance_initial = init_ops.constant_initializer(\
                moving_variance_value,
                dtypes.float32)
      moving_variance = gap_finetune.get_variable(\
                name='moving_variance',
                shape=moving_variance_value.shape,
                dtype=dtypes.float32,
                initializer=moving_variance_initial,
                gap=self.gap,
                gap_vars=self.gap_vars)

      def mean_var_with_update():
        mean, variance = nn_impl.moments(input_x, [0, 1, 2], name='moments')
        with ops.control_dependencies([\
            moving_averages.assign_moving_average(\
                          moving_mean, mean, 0.9),
            moving_averages.assign_moving_average(\
                          moving_variance, variance, 0.9)]):
          return array_ops.identity(mean), array_ops.identity(variance)

      mean, variance = control_flow_ops.cond(is_training, \
                            mean_var_with_update, \
                            lambda: (moving_mean, moving_variance))

      out = nn_impl.batch_normalization(input_x,
                                        mean,
                                        variance,
                                        beta,
                                        gamma,
                                        0.001)
      return out

  def _build_net(self):
    """ Build the network for test. """
    # network define
    x = constant_op.constant(1.0, shape=[1, 28, 28, 3], dtype=dtypes.float32)
    is_training = ops.convert_to_tensor(True)

    # conv1
    x = self._conv2d(x, 'conv1', self.conv1_kernel, self.conv1_bias)
    x = self._batchnorm(x, 'bn1', self.bn1_gamma, self.bn1_beta, \
          self.bn1_moving_mean, self.bn1_moving_variance, \
          is_training)
    x = nn_ops.relu(x)

    # conv2a
    residual_x = self._conv2d(x,
                              'conv2a_1',
                              self.conv2a_1_kernel,
                              self.conv2a_1_bias)
    residual_x = self._batchnorm(residual_x,
                                 'bn2a_1',
                                 self.bn2a_1_gamma,
                                 self.bn2a_1_beta,
                                 self.bn2a_1_moving_mean,
                                 self.bn2a_1_moving_variance,
                                 is_training)
    residual_x = nn_ops.relu(residual_x)

    card_in = gap_finetune.split(residual_x,
                                 num_or_size_splits=4,
                                 axis=-1,
                                 gap=self.gap,
                                 gap_vars=self.gap_vars)
    card_out = []
    for i in range(4):
      out = self._conv2d(card_in[i],
                         'conv2a_2_%d'%i,
                         self.conv2a_2_kernel[i],
                         self.conv2a_2_bias[i])
      card_out.append(out)
    residual_x = array_ops.concat(card_out, axis=-1)
    residual_x = self._batchnorm(residual_x,
                                 'bn2a_2',
                                 self.bn2a_2_gamma,
                                 self.bn2a_2_beta,
                                 self.bn2a_2_moving_mean,
                                 self.bn2a_2_moving_variance,
                                 is_training)
    residual_x = nn_ops.relu(residual_x)

    residual_x = self._conv2d(residual_x,
                              'conv2a_3',
                              self.conv2a_3_kernel,
                              self.conv2a_3_bias)
    residual_x = self._batchnorm(residual_x,
                                 'bn2a_3',
                                 self.bn2a_3_gamma,
                                 self.bn2a_3_beta,
                                 self.bn2a_3_moving_mean,
                                 self.bn2a_3_moving_variance,
                                 is_training)

    # conv2b
    shortcut_x = self._conv2d(x, 'conv2b', self.conv2b_kernel, self.conv2b_bias)
    shortcut_x = self._batchnorm(shortcut_x,
                                 'bn2b',
                                 self.bn2b_gamma,
                                 self.bn2b_beta,
                                 self.bn2b_moving_mean,
                                 self.bn2b_moving_variance,
                                 is_training)

    x = nn_ops.relu(residual_x + shortcut_x)

    # conv3
    x = self._conv2d(x, 'conv3', self.conv3_kernel, self.conv3_bias)
    x = self._batchnorm(x, 'bn3', self.bn3_gamma, self.bn3_beta, \
          self.bn3_moving_mean, self.bn3_moving_variance,\
          is_training)
    conv_output = nn_ops.relu(x)

    # loss
    self.loss = nn_ops.l2_loss(conv_output)

  def _get_l1_loss_gt(self):
    # l1_loss ground truth
    loss_l1_gt = 0.0
    loss_l1_gt += np.sum(np.abs(self.bn1_gamma))
    loss_l1_gt += np.sum(np.abs(self.bn2a_1_gamma))
    loss_l1_gt += np.sum(np.abs(self.bn2a_2_gamma))
    loss_l1_gt += np.sum(np.abs(self.bn2a_3_gamma))
    loss_l1_gt += np.sum(np.abs(self.bn2b_gamma))
    loss_l1_gt += np.sum(np.abs(self.bn3_gamma))
    return loss_l1_gt

  def _get_data_dict(self):
    data_dict = dict()
    for var in variables.global_variables():
      name = var.name.split(':')[0]
      data_dict[name] = var.eval()
    return data_dict

  def _gt_get_threshold(self, data, slim_ratio=0.5):
    """ Function to get the pruning threshold given `slim_ratio' """
    gammas = []
    for name, value in data.items():
      if 'Momentum' in name or 'local_step' in name or 'biased' in name:
        continue
      if 'gamma' not in name:
        continue
      gammas = np.hstack((gammas, value))
    gammas.sort()
    thres = gammas[int(slim_ratio*len(gammas))]
    return thres

  def _gt_get_mask(self, data, thres):
    """ Function to get the pruning mask given the threshold `thres'. """
    masks = dict()
    for name, value in data.items():
      if 'Momentum' in name or 'local_step' in name or 'biased' in name:
        continue
      if 'gamma' not in name:
        continue

      mask = value >= thres
      if np.sum(mask) == 0:
        inx = np.argsort(-gamma)
        min_len = int(max(0.1*len(gamma), 1))
        mask[inx[:min_len]] = True

      if 'bn2a_2' in name or 'bn2a_1' in name:
        mask_cards = np.split(mask, 4, axis=-1)
        value_cards = np.split(value, 4, axis=-1)
        for i in range(4):
          if np.sum(mask_cards[i]) == 0:
            inx = np.argsort(-np.array(value_cards[i]))
            min_len = int(max(0.1*len(value_cards[i]), 1))
            mask_cards[i][inx[:min_len]] = True
        masks[name] = np.concatenate(mask_cards, axis=-1)
      else:
        masks[name] = mask

    # two path of add: logical_or
    updated_mask = np.logical_or(masks['bn2a_3/gamma'], masks['bn2b/gamma'])
    masks['bn2a_3/gamma'] = updated_mask
    masks['bn2b/gamma'] = updated_mask

    # set the ending gamma to all true
    masks['bn3/gamma'] = np.logical_or(masks['bn3/gamma'], True)

    # convert numpy to list
    for key, value in masks.items():
      masks[key] = list(value)

    return masks

  def _gt_get_pruned_data(self, data, masks):
    """ Function to get the pruned variables given the original data dict `data' and pruning mask.
    Args:
      data: orginal data dict structured by
          {key: variable name, value: weight value}
      masks: pruning mask structured by
          {key: gamma name, value: mask for the corresponding gamma variable}
    Return:
      pruned_data: pruned data dict contains
        1) standart variable: {key: variable name, value: pruned weight value}
        2) split shape structured by
            { key: split scope name+`/split',
              value: split_subdict}
            and the `split_subdict' is structured by:
                { key: index,
                  value: dict{'Tag': False, 'Value': split output shape}}
           --> split_subdict with keys `index' mainly to
               process multiple `Split' opertaion within the same scope
    """
    pruned_data = dict()
    for name, value in data.items():
      if 'Momentum' in name \
          or 'local_step' in name \
          or 'biased' in name \
          or 'global_step' in name:
        continue
      # conv1, bn1
      if 'conv1' in name or 'bn1' in name:
        output_mask = masks['bn1/gamma']
        if 'kernel' in name:
          pruned_value = value[:, :, :, output_mask]
        else:
          pruned_value = value[output_mask]
        pruned_data[name] = pruned_value
      # conv2a_1, bn2a_1
      elif '2a_1' in name:
        input_mask = masks['bn1/gamma']
        output_mask = masks['bn2a_1/gamma']
        if 'kernel' in name:
          pruned_value = value[:, :, input_mask, :]
          pruned_value = pruned_value[:, :, :, output_mask]
        else:
          pruned_value = value[output_mask]
        pruned_data[name] = pruned_value
      # conv2a_2
      elif '2a_2' in name:
        input_mask = np.array(masks['bn2a_1/gamma'])
        output_mask = np.array(masks['bn2a_2/gamma'])
        if 'kernel' in name or 'bias' in name:
          card_inx = int(name.split('/')[0].split('_')[-1])
          input_masks = np.split(input_mask, 4, axis=-1)
          output_masks = np.split(output_mask, 4, axis=-1)
          input_mask = input_masks[card_inx]
          output_mask = output_masks[card_inx]
          if 'kernel' in name:
            pruned_value = value[:, :, input_mask, :]
            pruned_value = pruned_value[:, :, :, output_mask]
          else:
            pruned_value = value[output_mask]
        else:
          pruned_value = value[output_mask]
        pruned_data[name] = pruned_value
      # conv2a_3
      elif '2a_3' in name:
        input_mask = masks['bn2a_2/gamma']
        output_mask = masks['bn2a_3/gamma']
        if 'kernel' in name:
          pruned_value = value[:, :, input_mask, :]
          pruned_value = pruned_value[:, :, :, output_mask]
        else:
          pruned_value = value[output_mask]
        pruned_data[name] = pruned_value
      # conv2b
      elif '2b' in name:
        input_mask = masks['bn1/gamma']
        output_mask = masks['bn2b/gamma']
        if 'kernel' in name:
          pruned_value = value[:, :, input_mask, :]
          pruned_value = pruned_value[:, :, :, output_mask]
        else:
          pruned_value = value[output_mask]
        pruned_data[name] = pruned_value
      # conv3
      elif 'conv3' in name or 'bn3' in name:
        input_mask = masks['bn2b/gamma']
        output_mask = masks['bn3/gamma']
        if 'kernel' in name:
          pruned_value = value[:, :, input_mask, :]
          pruned_value = pruned_value[:, :, :, output_mask]
        else:
          pruned_value = value[output_mask]
        pruned_data[name] = pruned_value

    split_shape = []
    split_mask = np.array(masks['bn2a_1/gamma'])
    mask_cards = np.split(split_mask, 4, axis=-1)
    for i in range(4):
      split_shape.append(np.sum(mask_cards[i]))
    pruned_data['split'] = dict()
    pruned_data['split'][0] = {\
          'Tag': False,
          'Value': split_shape}
    return pruned_data

  def testGAPstep1(self):
    """ Test function for step1 in GAP. """
    with ops.Graph().as_default():
      self.gap = False
      self.gap_vars = None
      self._build_net()
      # gap: add l1_loss
      key_graph = gap_train_prune.KeyGraph(ops.get_default_graph())
      l1_loss = key_graph.gap_l1_loss()

      # run network
      with self.test_session(\
            graph=ops.get_default_graph(),
            use_gpu=True) as sess:
        sess.run(variables.global_variables_initializer())
        loss_l1_gap = sess.run(l1_loss)

      loss_l1_gt = self._get_l1_loss_gt()
      self.assertAllClose(loss_l1_gap, loss_l1_gt, rtol=1e-4)

  def testGAPstep2(self):
    """ Test function for step2 in GAP. """
    with ops.Graph().as_default():
      self.gap = False
      self.gap_vars = None
      self._build_net()
      # gap: build key graph
      key_graph = gap_train_prune.KeyGraph(ops.get_default_graph())
      key_graph.gap_init()

      with self.test_session(\
            graph=ops.get_default_graph(),
            use_gpu=True) as sess:
        sess.run(variables.global_variables_initializer())
        data = self._get_data_dict()

      slim_ratio = 0.5
      # ground truth
      thres_gt = self._gt_get_threshold(data, slim_ratio)
      masks_gt = self._gt_get_mask(data, thres_gt)
      pruned_data_gt = self._gt_get_pruned_data(data, masks_gt)
      # gap
      masks_gap = key_graph.gap_get_mask(data, slim_ratio=slim_ratio)
      pruned_data_gap = key_graph.gap_get_pruned_variables(data, masks_gap)

      for name, mask_gt in masks_gt.items():
        self.assertTrue(masks_gap.has_key(name))
        mask_gap = masks_gap[name]
        self.assertAllClose(mask_gap, mask_gt, rtol=1e-4)

      for name, value_gt in pruned_data_gt.items():
        self.assertTrue(pruned_data_gap.has_key(name))
        value_gap = pruned_data_gap[name]
        if not isinstance(value_gt, dict):
          self.assertAllClose(value_gap, value_gt, rtol=1e-4)
        else:
          for inx, split_value_gt in value_gt.items():
            self.assertTrue(value_gap.has_key(inx))
            split_value_gap = value_gap[inx]
            self.assertAllClose(\
                  split_value_gap['Tag'],
                  split_value_gt['Tag'],
                  rtol=1e-4)
            self.assertAllClose(\
                  split_value_gap['Value'],
                  split_value_gt['Value'],
                  rtol=1e-4)

  def testGAPstep3(self):
    """ Test function for step3 in GAP. """
    # generate the original data_dict
    with ops.Graph().as_default():
      self.gap = False
      self.gap_vars = None
      self._build_net()

      with self.test_session(\
          graph=ops.get_default_graph(),
          use_gpu=True) as sess:
        sess.run(variables.global_variables_initializer())
        original_loss = sess.run(self.loss)
        data = self._get_data_dict()

      slim_ratio = 0.0
      # ground truth
      thres_gt = self._gt_get_threshold(data, slim_ratio)
      masks_gt = self._gt_get_mask(data, thres_gt)
      pruned_data_gt = self._gt_get_pruned_data(data, masks_gt)

    with ops.Graph().as_default():
      self.gap = True
      self.gap_vars = pruned_data_gt
      #with self.assertRaisesOpError("GAP build finetune model failed."):
      self._build_net()
      with self.test_session(\
          graph=ops.get_default_graph(),
          use_gpu=True) as sess:
        sess.run(variables.global_variables_initializer())
        finetune_loss = sess.run(self.loss)

      self.assertAllClose(original_loss, finetune_loss, rtol=1e-4)

if __name__ == "__main__":
  test.main()
