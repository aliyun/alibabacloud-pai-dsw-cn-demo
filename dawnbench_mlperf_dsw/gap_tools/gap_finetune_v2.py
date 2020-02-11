"""
   Code for implementation of variables and split functions in gap finetuning.
   Usage document ref to https://lark.alipay.com/pai/developer-docs/gap_tools
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import init_ops
import pickle

class GapFinetune(object):
  def __init__(self, pruned_var_path=None):
    try:
      with open(pruned_var_path, 'rb') as f:
        #self.gap_vars = pickle.load(f, encoding='latin1')
        self.gap_vars = pickle.load(f)
    except:
      raise ValueError("Error when load the pruned variables from %s!" % pruned_var_path)

  def get_variable(self, getter,
                   name,
                   shape=None,
                   dtype=None,
                   initializer=None,
                   *args, **kwargs):
    """ This function create a variable using the pruned values in gap
  
    args:
     name: The name of the new or existing variable.
     shape: Shape of the new or existing variable.
     dtype: Type of the new or existing variable (defaults to `DT_FLOaT`).
     initializer: Initializer for the variable if one is created.
  
    Returns:
     The created or existing `Variable` (or `partitionedVariable`, if a
     partitioner was used).
    Raises:
     ValueError: when creating a new variable and shape is not declared,
     when violating reuse during variable creation, or when `initializer` dtype,
     when `dtype` don't match. Reuse is set inside `variable_scope`,
     when `gap_vars' does not have the key of the variable name (under scope)
    """
  
    if name in self.gap_vars:
      value = self.gap_vars[name]
      shape = value.shape
      initializer = init_ops.constant_initializer(value, dtype)
  
    return getter(\
        name=name,
        shape=shape,
        initializer=initializer,
        *args, **kwargs)


