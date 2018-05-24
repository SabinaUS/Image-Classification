from layers import *
import tensorflow as tf

class ResNet():
    def __init__(self, num_blocks_each_layer):
        if (len(num_blocks_each_layer) != 4):
            raise ValueError('num_blocks_each_layer should be length 4!')
        self.num_basic_blocks_each_layer = num_blocks_each_layer

    def _make_basic_block(self, input_layer, input_channels, output_channels, scope, stride=1):
        with tf.variable_scope(scope):
            out = conv2d(input_layer, input_channels, output_channels, 'conv2d_0', stride=stride) 
            out = batch_norm(out, 'bn_0')   
            out = lrelu(out)
            out = conv2d(out, output_channels, output_channels, 'conv2d_1') 
            out = batch_norm(out, 'bn_1')   
            
            if stride != 1 or input_channels != output_channels:
               input_layer = conv2d(input_layer, input_channels, output_channels, 'conv2d_2', stride=stride, kernel_size=1)
               input_layer = batch_norm(input_layer, 'bn_2')

            out = tf.add(out, input_layer)
            out = lrelu(out)
            return out
            
    def _make_layer(self, input_layer, input_channels, output_channels, num_blocks, scope, stride=1):
        with tf.variable_scope(scope):
            out = self._make_basic_block(input_layer, input_channels, output_channels, 'basic_block_0', stride=stride)
            for i in range(1, num_blocks):
                out = self._make_basic_block(out, output_channels, output_channels, 'basic_block' + str(i))
            return out
             
    def build_network(self, input_images, num_classes):
        with tf.variable_scope('ResNet'):

            out = conv2d(input_images, 3, 64, 'conv2d_0', kernel_size=7, stride=2)
            out = batch_norm(out, 'bn_0')
            out = lrelu(out)
            out = tf.nn.max_pool(out, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

            out = self._make_layer(out, 64, 64, self.num_basic_blocks_each_layer[0], 'layer_1', stride=1)
            out = self._make_layer(out, 64, 128, self.num_basic_blocks_each_layer[1], 'layer_2', stride=2)
            out = self._make_layer(out, 128, 256, self.num_basic_blocks_each_layer[2], 'layer_3', stride=2)
            out = self._make_layer(out, 256, 512, self.num_basic_blocks_each_layer[3], 'layer_4', stride=2)

            out = tf.nn.avg_pool(out, [1, 7, 7, 1], [1, 1, 1, 1], 'SAME')
            out = tf.reshape(out, [tf.shape(out)[0], -1])
            out = fully_connected('fc', out, 512, num_classes)
            
            return out
