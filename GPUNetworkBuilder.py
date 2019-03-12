import tensorflow as tf
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import init_ops

from collections import defaultdict

class GPUNetworkBuilder(object):
    """This class provides convenient methods for constructing feed-forward
    networks with internal data layout of 'NCHW'.
    """
    def __init__(self,
                 is_training,
                 dtype=tf.float32,
                 activation='RELU',
                 use_batch_norm=True,
                 batch_norm_config = {'decay':   0.9,
                                      'epsilon': 1e-4,
                                      'scale':   True,
                                      'zero_debias_moving_mean': False}):
        self.dtype             = dtype
        self.activation_func   = activation
        self.is_training       = is_training
        self.use_batch_norm    = use_batch_norm
        self.batch_norm_config = batch_norm_config
        self._layer_counts     = defaultdict(lambda: 0)
    def _count_layer(self, layer_type):
        idx  = self._layer_counts[layer_type]
        name = layer_type + str(idx)
        self._layer_counts[layer_type] += 1
        return name
    def _get_variable(self, name, shape, dtype=None,
                      initializer=None, seed=None):
        if dtype is None:
            dtype = self.dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        elif (isinstance(initializer, float) or
              isinstance(initializer, int)):
            initializer = tf.constant_initializer(float(initializer))
        return tf.get_variable(name, shape, dtype, initializer)
    def _to_nhwc(self, x):
        return tf.transpose(x, [0,2,3,1])
    def _from_nhwc(self, x):
        return tf.transpose(x, [0,3,1,2])
    def _bias(self, input_layer):
        num_outputs = input_layer.get_shape().as_list()[1]
        biases = self._get_variable('biases', [num_outputs], input_layer.dtype,
                                    initializer=0)
        if len(input_layer.get_shape()) == 4:
            return tf.nn.bias_add(input_layer, biases,
                                  data_format='NCHW')
        else:
            return input_layer + biases
    def _batch_norm(self, input_layer, scope):
        return tf.contrib.layers.batch_norm(input_layer,
                                            is_training=self.is_training,
                                            scope=scope,
                                            data_format='NCHW',
                                            fused=True,
                                            **self.batch_norm_config)
    def _bias_or_batch_norm(self, input_layer, scope, use_batch_norm):
        if use_batch_norm is None:
            use_batch_norm = self.use_batch_norm
        if use_batch_norm:
            return self._batch_norm(input_layer, scope)
        else:
            return self._bias(input_layer)
    def input_layer(self, input_layer):
        """Converts input data into the internal format"""
        x = self._from_nhwc(input_layer)
        x = tf.cast(x, self.dtype)
        # Rescale and shift to [-1,1]
        x = x * (1./127.5) - 1
        return x
    def conv(self, input_layer, num_filters, filter_size,
             filter_strides=(1,1), padding='SAME',
             activation=None, use_batch_norm=None):
        """Applies a 2D convolution layer that includes bias or batch-norm
        and an activation function.
        """
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_inputs, num_filters]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('conv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            if padding == 'SAME_RESNET': # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                padding = [[0, 0], [0, 0],
                           [pad_beg, pad_end], [pad_beg, pad_end]]
                input_layer = tf.pad(input_layer, padding)
                padding = 'VALID'
            x = tf.nn.conv2d(input_layer, kernel, strides,
                             padding=padding, data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x
    def deconv(self, input_layer, num_filters, filter_size,
               filter_strides=(2,2), padding='SAME',
               activation=None, use_batch_norm=None):
        """Applies a 'transposed convolution' layer that includes bias or
        batch-norm and an activation function.
        """
        num_inputs  = input_layer.get_shape().as_list()[1]
        ih, iw      = input_layer.get_shape().as_list()[2:]
        output_shape = [-1, num_filters,
                        ih*filter_strides[0], iw*filter_strides[1]]
        kernel_shape = [filter_size[0], filter_size[1],
                        num_filters, num_inputs]
        strides = [1, 1, filter_strides[0], filter_strides[1]]
        with tf.variable_scope(self._count_layer('deconv')) as scope:
            kernel = self._get_variable('weights', kernel_shape,
                                        input_layer.dtype)
            x = tf.nn.conv2d_transpose(input_layer, kernel, output_shape,
                                       strides, padding=padding,
                                       data_format='NCHW')
            x = self._bias_or_batch_norm(x, scope, use_batch_norm)
            x = self.activate(x, activation)
            return x
    def activate(self, input_layer, funcname=None):
        """Applies an activation function"""
        if isinstance(funcname, tuple):
            funcname = funcname[0]
            params = funcname[1:]
        if funcname is None:
            funcname = self.activation_func
        if funcname == 'LINEAR':
            return input_layer
        activation_map = {
            'RELU':    tf.nn.relu,
            'RELU6':   tf.nn.relu6,
            'ELU':     tf.nn.elu,
            'SIGMOID': tf.nn.sigmoid,
            'TANH':    tf.nn.tanh,
            'LRELU':   lambda x, name: tf.maximum(params[0]*x, x, name=name)
        }
        return activation_map[funcname](input_layer, name=funcname.lower())
    def pool(self, input_layer, funcname, window_size,
                 window_strides=(2,2),
                 padding='VALID'):
        """Applies spatial pooling"""
        pool_map = {
            'MAX': tf.nn.max_pool,
            'AVG': tf.nn.avg_pool
        }
        kernel_size    = [1, 1, window_size[0], window_size[1]]
        kernel_strides = [1, 1, window_strides[0], window_strides[1]]
        return pool_map[funcname](input_layer, kernel_size, kernel_strides,
                                  padding, data_format='NCHW',
                                  name=funcname.lower())
    def project(self, input_layer, num_outputs, height, width,
                activation=None):
        """Linearly projects to an image-like tensor"""
        with tf.variable_scope(self._count_layer('project')):
            x = self.fully_connected(input_layer, num_outputs*height*width,
                                     activation=activation)
            x = tf.reshape(x, [-1, num_outputs, height, width])
            return x
    def flatten(self, input_layer):
        """Flattens the spatial and channel dims into a single dim (4D->2D)"""
        # Note: This ensures the output order matches that of NHWC networks
        input_layer = self._to_nhwc(input_layer)
        input_shape = input_layer.get_shape().as_list()
        num_inputs  = input_shape[1]*input_shape[2]*input_shape[3]
        return tf.reshape(input_layer, [-1, num_inputs], name='flatten')
    def spatial_avg(self, input_layer):
        """Averages over spatial dimensions (4D->2D)"""
        return tf.reduce_mean(input_layer, [2, 3], name='spatial_avg')
    def fully_connected(self, input_layer, num_outputs, activation=None):
        """Applies a fully-connected set of weights"""
        num_inputs = input_layer.get_shape().as_list()[1]
        kernel_size = [num_inputs, num_outputs]
        with tf.variable_scope(self._count_layer('fully_connected')):
            kernel = self._get_variable('weights', kernel_size,
                                        input_layer.dtype)
            x = tf.matmul(input_layer, kernel)
            x = self._bias(x)
            x = self.activate(x, activation)
            return x
    def inception_module(self, input_layer, name, cols):
        """Applies an inception module with a given form"""
        with tf.name_scope(name):
            col_layers      = []
            col_layer_sizes = []
            for c, col in enumerate(cols):
                col_layers.append([])
                col_layer_sizes.append([])
                x = input_layer
                for l, layer in enumerate(col):
                    ltype, args = layer[0], layer[1:]
                    if   ltype == 'conv': x = self.conv(x, *args)
                    elif ltype == 'pool': x = self.pool(x, *args)
                    elif ltype == 'share':
                        # Share matching layer from previous column
                        x = col_layers[c-1][l]
                    else: raise KeyError("Invalid layer type for " +
                                         "inception module: '%s'" % ltype)
                    col_layers[c].append(x)
            catdim  = 1
            catvals = [layers[-1] for layers in col_layers]
            x = tf.concat(catvals, catdim)
            return x
    def residual(self, input_layer, net, scale=1.0, activation='RELU'):
        """Applies a residual layer"""
        input_size     = input_layer.get_shape().as_list()
        num_inputs     = input_size[1]
        output_layer   = scale*net(self, input_layer)
        output_size    = output_layer.get_shape().as_list()
        num_outputs    = output_size[1]
        kernel_strides = (input_size[2]//output_size[2],
                          input_size[3]//output_size[3])
        with tf.name_scope('residual'):
            if (num_outputs != num_inputs or
                kernel_strides[0] != 1 or
                kernel_strides[1] != 1):
                input_layer = self.conv(input_layer, num_outputs, [1, 1],
                                        kernel_strides, activation='LINEAR')
            x = self.activate(input_layer + output_layer, activation)
            return x
    def dropout(self, input_layer, keep_prob=0.5):
        """Applies a dropout layer if is_training"""
        if self.is_training:
            dtype = input_layer.dtype
            with tf.variable_scope(self._count_layer('dropout')):
                keep_prob_tensor = tf.constant(keep_prob, dtype=dtype)
                return tf.nn.dropout(input_layer, keep_prob_tensor)
        else:
            return input_layer