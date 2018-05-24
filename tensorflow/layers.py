import tensorflow as tf

def conv2d(input_layer, input_channels, output_channels, scope, kernel_size=3, stride=1, padding='SAME', bias=False):

    strides = [1, stride, stride, 1]
    with tf.variable_scope(scope):
        conv_filter = tf.get_variable(
            'weight',
            shape = [kernel_size, kernel_size, input_channels, output_channels],
            dtype = tf.float32,
            initializer = tf.contrib.layers.variance_scaling_initializer(),
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)
        )
        conv = tf.nn.conv2d(input_layer, conv_filter, strides, padding)

        if bias:
            bias = tf.get_variable(
                'bias',
                shape = [output_channels],
                dtype = tf.float32,
                initializer = tf.constant_initializer(0.0)
            )
            output_layer = tf.nn.bias_add(conv, bias)
            output_layer = tf.reshape(output_layer, conv.get_shape())
        else:
            output_layer = conv

        return output_layer

def batch_norm(input_layer, scope):
    output_layer = tf.contrib.layers.batch_norm(
        input_layer,
        decay = 0.9,
        scale = True,
        epsilon = 1e-5,
        is_training = True,
        scope = scope
    )
    return output_layer

def lrelu(input_layer, leak=0.2):
    output_layer = tf.nn.relu(input_layer)
    return output_layer

def fully_connected(scope, input_layer, input_dim, output_dim):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        fc_weight = tf.get_variable(
            'fc_weight',
            shape = [input_dim, output_dim],
            dtype = tf.float32,
            initializer = tf.contrib.layers.variance_scaling_initializer(),
            regularizer = tf.contrib.layers.l2_regularizer(scale=0.0002)            
        )

        fc_bias = tf.get_variable(
            'fc_bias',
            shape = [output_dim],
            dtype = tf.float32,
            initializer = tf.constant_initializer(0.0)
        )

        output_layer = tf.matmul(input_layer, fc_weight) + fc_bias

        return output_layer


