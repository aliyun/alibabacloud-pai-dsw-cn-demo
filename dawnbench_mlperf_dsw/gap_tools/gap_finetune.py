"""
   Code for implementation of variables and split functions in gap finetuning.
   Usage document ref to https://lark.alipay.com/pai/developer-docs/gap_tools
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables

def get_variable(name,
                 shape=None,
                 dtype=None,
                 initializer=None,
                 regularizer=None,
                 trainable=True,
                 collections=None,
                 caching_device=None,
                 partitioner=None,
                 validate_shape=True,
                 use_resource=None,
                 custom_getter=None,
                 gap=False,
                 gap_vars=None):

  """ This function create a variable using the pruned values in gap

  if the option `gap' is closed, it falls back to standard get_variable,
  if the option `gap' is open,
    it create a variable initialized by the pruned value provided by `gap_vars',
    the original option `initializer' and `shape' is ignored

  args:
   name: The name of the new or existing variable.
   shape: Shape of the new or existing variable.
   dtype: Type of the new or existing variable (defaults to `DT_FLOaT`).
   initializer: Initializer for the variable if one is created.
   regularizer: a (Tensor -> Tensor or None) function; the result of
    applying it on a newly created variable will be added to the collection
    @{tf.graphKeys.REgULaRIZaTION_LOSSES} and can be used for regularization.
   %scollections: List of graph collections keys to add the Variable to.
    Defaults to `[%s]` (see `tf.Variable`).
   caching_device: Optional device string or function describing where the
    Variable should be cached for reading.  Defaults to the Variable's
    device.  If not `None`, caches on another device.  Typical use is to
    cache on the device where the Ops using the Variable reside, to
    deduplicate copying through `Switch` and other conditional statements.
   partitioner: Optional callable that accepts a fully defined `TensorShape`
    and `dtype` of the Variable to be created, and returns a list of
    partitions for each axis (currently only one axis can be partitioned).
   validate_shape: If False, allows the variable to be initialized with a
     value of unknown shape. If True, the default, the shape of initial_value
     must be known.
   use_resource: If False, creates a regular Variable. If true, creates an
    experimental ResourceVariable instead with well-defined semantics.
    Defaults to False (will later change to True). In Eager mode, this argument
    is always forced to be True.
   custom_getter: Callable that takes as a first argument the true getter, and
    allows overwriting the internal get_variable method.
    The signature of `custom_getter` should match that of this method,
    but the most future-proof version will allow for changes:
    `def custom_getter(getter, *args, **kwargs)`.  Direct access to
    all `get_variable` parameters is also allowed:
    `def custom_getter(getter, name, *args, **kwargs)`.  a simple identity
    custom getter that simply creates variables with modified names is:
    ```python
    def custom_getter(getter, name, *args, **kwargs):
     return getter(name + '_suffix', *args, **kwargs)
    ```
   gap: bool option,
     if open (open), initialize the variable using the value in gap_vars,
     otherwise `initializer' and `shape' is used for the variable initialization
   gap_vars: dict of values, with the key being the variable name,
     and the value denoting the value in numpy array format

  Returns:
   The created or existing `Variable` (or `partitionedVariable`, if a
   partitioner was used).
  Raises:
   ValueError: when creating a new variable and shape is not declared,
   when violating reuse during variable creation, or when `initializer` dtype,
   when `dtype` don't match. Reuse is set inside `variable_scope`,
   when `gap' is open and `gap_vars' is not provided
   and `gap_vars' does not have the key of the variable name (under scope)
  """

  if gap is True:
    if gap_vars is None:
      raise ValueError("No gap variable dict 'gap_vars' provided!")
    scope = variable_scope.get_variable_scope()
    if scope.name == "":
      var_name = name
    else:
      var_name = scope.name + '/' + name

    if not gap_vars.has_key(var_name):
      raise ValueError("Wrong variable name to lookup in 'gap_vars': %s" \
          % var_name)
    value = gap_vars[var_name]
    if dtype is None:
      initializer = init_ops.constant_initializer(value)
    else:
      initializer = init_ops.constant_initializer(value, dtype)

    shape = value.shape

  return variable_scope.get_variable(\
      name=name,
      shape=shape,
      dtype=dtype,
      initializer=initializer,
      regularizer=regularizer,
      trainable=trainable,
      collections=collections,
      caching_device=caching_device,
      partitioner=partitioner,
      validate_shape=validate_shape,
      use_resource=use_resource,
      custom_getter=custom_getter)


def Variable(initial_value=None,
             trainable=True,
             collections=None,
             validate_shape=True,
             caching_device=None,
             name=None,
             variable_def=None,
             dtype=None,
             expected_shape=None,
             import_scope=None,
             constraint=None,
             gap=False,
             gap_vars=None):

  """Creates a new variable with value `initial_value` in gap.

  if the option `gap' is closed, it falls back to standard Variable,
  if the option `gap' is open,
    it create a variable initialized by the pruned value provided by `gap_vars',
    the original option `initial_value' and `expected_shape' is ignored

  The new variable is added to the graph collections listed in `collections`,
  which defaults to `[graphKeys.gLOBaL_VaRIaBLES]`.

  If `trainable` is `True` the variable is also added to the graph collection
  `graphKeys.TRaINaBLE_VaRIaBLES`.

  This constructor creates both a `variable` Op and an `assign` Op to set the
  variable to its initial value.

  args:
   initial_value: a `Tensor`, or python object convertible to a `Tensor`,
    which is the initial value for the Variable. The initial value must have
    a shape specified unless `validate_shape` is set to False. Can also be a
    callable with no argument that returns the initial value when called. In
    that case, `dtype` must be specified. (Note that initializer functions
    from init_ops.py must first be bound to a shape before being used here.)
   trainable: If `True`, the default, also adds the variable to the graph
    collection `graphKeys.TRaINaBLE_VaRIaBLES`. This collection is used as
    the default list of variables to use by the `Optimizer` classes.
   collections: List of graph collections keys. The new variable is added to
    these collections. Defaults to `[graphKeys.gLOBaL_VaRIaBLES]`.
   validate_shape: If `False`, allows the variable to be initialized with a
    value of unknown shape. If `True`, the default, the shape of
    `initial_value` must be known.
   caching_device: Optional device string describing where the Variable
    should be cached for reading.  Defaults to the Variable's device.
    If not `None`, caches on another device.  Typical use is to cache
    on the device where the Ops using the Variable reside, to deduplicate
    copying through `Switch` and other conditional statements.
   name: Optional name for the variable. Defaults to `'Variable'` and gets
    uniquified automatically.
   variable_def: `VariableDef` protocol buffer. If not `None`, recreates
    the Variable object with its contents, referencing the variable's nodes
    in the graph, which must already exist. The graph is not changed.
    `variable_def` and the other arguments are mutually exclusive.
   dtype: If set, initial_value will be converted to the given type.
    If `None`, either the datatype will be kept (if `initial_value` is
    a Tensor), or `convert_to_tensor` will decide.
   expected_shape: a TensorShape. If set, initial_value is expected
    to have this shape.
   import_scope: Optional `string`. Name scope to add to the
    `Variable.` Only used when initializing from protocol buffer.
   constraint: an optional projection function to be applied to the variable
    after being updated by an `Optimizer` (e.g. used to implement norm
    constraints or value constraints for layer weights). The function must
    take as input the unprojected Tensor representing the value of the
    variable and return the Tensor for the projected value
    (which must have the same shape). Constraints are not safe to
    use when doing asynchronous distributed training.
   gap: bool option,
     if open (open), initialize the variable using the value in gap_vars,
     otherwise `initial_value' and `expected_shape' is used
   gap_vars: dict of values,
     with the key being the variable name,
     and the value denoting the value in numpy array format

  Returns:
   The created `Variable`.

  Raises:
   ValueError: If both `variable_def` and initial_value are specified.
   ValueError: If the initial value is not specified, or does not have a
    shape and `validate_shape` is `True`.
   ValueError: If `gap' is open and `gap_vars' is not provided
   ValueError: If `gap_vars' does not have the key of the variable name
   RuntimeError: If created in EagER mode.
  """

  if gap is True:
    if gap_vars is None:
      raise ValueError("No gap variable dict 'gap_vars' provided!")
    scope = variable_scope.get_variable_scope()
    if scope.name == "":
      var_name = name
    else:
      var_name = scope.name + '/' + name

    if not gap_vars.has_key(var_name):
      raise ValueError("Wrong variable name to lookup in 'gap_vars': %s" \
          % var_name)

    initial_value = gap_vars[var_name]
    if variable_def:
      # If variable_def is provided, ignore it.
      variable_def = None

  return variables.Variable(
      initial_value=initial_value,
      trainable=trainable,
      collections=collections,
      validate_shape=validate_shape,
      caching_device=caching_device,
      name=name,
      variable_def=variable_def,
      dtype=dtype,
      expected_shape=expected_shape,
      import_scope=import_scope,
      constraint=constraint)

def split(value,
          num_or_size_splits,
          axis=0,
          num=None,
          name="split",
          gap=False,
          gap_vars=None):
  """ Split a tensor into sub tensors, using the pruned value shape in gap

  if the option `gap' is closed, it falls back to standard split,
  if the option `gap' is open,
    it splits the input tensor based on the shape provided by `gap_vars',
    the original option `num_or_size_splits' is ignored

  Args:
    value: The `Tensor` to split.
    num_or_size_splits: Either a 0-D integer `Tensor` indicating the number of
      splits along split_dim or a 1-D integer `Tensor` integer tensor containing
      the sizes of each output tensor along split_dim. If a scalar then it must
      evenly divide `value.shape[axis]`; otherwise the sum of sizes along the
      split dimension must match that of the `value`.
    axis: A 0-D `int32` `Tensor`. The dimension along which to split.
      Must be in the range `[-rank(value), rank(value))`. Defaults to 0.
    num: Optional, used to specify the number of outputs when it cannot be
      inferred from the shape of `size_splits`.
    name: A name for the operation (optional).

  Returns:
    if `num_or_size_splits` is a scalar returns `num_or_size_splits` `Tensor`
    objects; if `num_or_size_splits` is a 1-D Tensor returns
    `num_or_size_splits.get_shape[0]` `Tensor` objects resulting from splitting
    `value`.

  Raises:
    ValueError: If `num` is unspecified and cannot be inferred.
  """

  if gap:
    if gap_vars is None:
      raise ValueError("No gap variable dict 'gap_vars' provided!")
    scope = ops.get_default_graph().get_name_scope()
    if scope == '':
      split_name = 'split'
    else:
      split_name = scope + '/split'

    if not gap_vars.has_key(split_name):
      raise ValueError("Wrong variable name to lookup in 'gap_vars': %s" \
          % split_name)

    inx = 0
    while inx < len(gap_vars[split_name]):
      if not gap_vars[split_name][inx]['Tag']:
        break
      inx += 1

    num_or_size_splits = gap_vars[split_name][inx]['Value']
    gap_vars[split_name][inx]['Tag'] = True

  return array_ops.split(\
              value=value,
              num_or_size_splits=num_or_size_splits,
              axis=axis,
              num=num,
              name=name)

class GapFinetune(object):
  def __init__(self, gap_vars, random_init=False):
    self.gap_vars = gap_vars
    self.random_init = random_init

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

    if name not in self.gap_vars:
      raise ValueError("Variable name not in the pruned variables provided")

    value = self.gap_vars[name]
    if self.random_init:
        if 'kernel' in name:
            initializer = tf.variance_scaling_initializer(distribution="truncated_normal", dtype=dtype)
        elif 'gamma' in name:
            initializer = init_ops.constant_initializer(np.ones_like(value), dtype)
        elif 'beta' in name:
            initializer = init_ops.constant_initializer(np.zeros_like(value), dtype)
        else:
            initializer = init_ops.constant_initializer(value, dtype)
    else:
        initializer = init_ops.constant_initializer(value, dtype)

    return getter(\
        name=name,
        shape=value.shape,
        initializer=initializer,
        *args, **kwargs)
