import tensorflow as tf
from functools import reduce
from operator import mul


def _linear(xs,output_size,bias,bias_start=0., scope=None):
    with tf.variable_scope(scope or 'linear_layer'):
        x = tf.concat(xs,-1)
        input_size = x.get_shape()[-1]
        W = tf.get_variable('W', shape=[input_size,output_size],dtype=tf.float32,
                            )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],dtype=tf.float32,
                                   initializer=tf.constant_initializer(bias_start))
            # print("=====================================")
            # print("x: ", x.shape)
            # print("W: ", W.shape)
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        return out


def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        args = [args]

    flat_args = [flatten(arg, 1) for arg in args] # for dense layer [(-1, d)]
    if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)# for dense layer [(-1, d)]
                     for arg in flat_args]
    flat_out = _linear(flat_args, output_size, bias, bias_start=bias_start, scope=scope) # dense
    out = reconstruct(flat_out, args[0], 1) # ()
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_layer"):
        d = arg.get_shape()[-1]  # embedding dim
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
                       is_train=is_train)
        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train)
            prev = cur
        return cur


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    dim_reduced_keep = dim_reduced_keep or keep

    ref_shape = ref.get_shape().as_list() # original shape
    tensor_shape = tensor.get_shape().as_list() # current shape
    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))] #
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    return out


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter
