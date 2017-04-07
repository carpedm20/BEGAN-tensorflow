# Code from https://github.com/david-berthelot/tf_img_tech/blob/master/tfswag/layers.py
import numpy as N
import numpy.linalg as LA
import tensorflow as tf

__author__ = 'David Berthelot'


def unboxn(vin, n):
    """vin = (batch, h, w, depth), returns vout = (batch, n*h, n*w, depth), each pixel is duplicated."""
    s = tf.shape(vin)
    vout = tf.concat([vin] * (n ** 2), 0)  # Poor man's replacement for tf.tile (required for Adversarial Training support).
    vout = tf.reshape(vout, [s[0] * (n ** 2), s[1], s[2], s[3]])
    vout = tf.batch_to_space(vout, [[0, 0], [0, 0]], n)
    return vout


def boxn(vin, n):
    """vin = (batch, h, w, depth), returns vout = (batch, h//n, w//n, depth), each pixel is averaged."""
    if n == 1:
        return vin
    s = tf.shape(vin)
    vout = tf.reshape(vin, [s[0], s[1] // n, n, s[2] // n, n, s[3]])
    vout = tf.reduce_mean(vout, [2, 4])
    return vout


class LayerBase:
    pass


class LayerConv(LayerBase):
    def __init__(self, name, w, n, nl=lambda x, y: x + y, strides=(1, 1, 1, 1),
                 padding='SAME', conv=None, use_bias=True, data_format="NCHW"):
        """w = (wy, wx), n = (n_in, n_out)"""
        self.nl = nl
        self.strides = list(strides)
        self.padding = padding
        self.data_format = data_format
        with tf.name_scope(name):
            if conv is None:
                conv = tf.Variable(tf.truncated_normal([w[0], w[1], n[0], n[1]], stddev=0.01), name='conv')
            self.conv = conv
            self.bias = tf.Variable(tf.zeros([n[1]]), name='bias') if use_bias else 0

    def __call__(self, vin):
        return self.nl(tf.nn.conv2d(vin, self.conv, strides=self.strides,
                                    padding=self.padding, data_format=self.data_format), self.bias)

class LayerEncodeConvGrowLinear(LayerBase):
    def __init__(self, name, n, width, colors, depth, scales, nl=lambda x, y: x + y, data_format="NCHW"):
        with tf.variable_scope(name) as vs:
            encode = []
            nn = n
            for x in range(scales):
                cl = []
                for y in range(depth - 1):
                    cl.append(LayerConv('conv_%d_%d' % (x, y), [width, width],
                                        [nn, nn], nl, data_format=data_format))
                cl.append(LayerConv('conv_%d_%d' % (x, depth - 1), [width, width],
                                    [nn, nn + n], nl, strides=[1, 2, 2, 1], data_format=data_format))
                encode.append(cl)
                nn += n
            self.encode = [LayerConv('conv_pre', [width, width], [colors, n], nl, data_format=data_format), encode]
        self.variables = tf.contrib.framework.get_variables(vs)

    def __call__(self, vin, carry=0, train=True):
        vout = self.encode[0](vin)
        for convs in self.encode[1]:
            for conv in convs[:-1]:
                vtmp = tf.nn.elu(conv(vout))
                vout = carry * vout + (1 - carry) * vtmp
            vout = convs[-1](vout)
        return vout, self.variables


class LayerDecodeConvBlend(LayerBase):
    def __init__(self, name, n, width, colors, depth, scales, nl=lambda x, y: x + y, data_format="NCHW"):
        with tf.variable_scope(name) as vs:
            decode = []
            for x in range(scales):
                cl = []
                n2 = 2 * n if x else n
                cl.append(LayerConv('conv_%d_%d' % (x, 0), [width, width],
                                    [n2, n], nl, data_format=data_format))
                for y in range(1, depth):
                    cl.append(LayerConv('conv_%d_%d' % (x, y), [width, width], [n, n], nl, data_format=data_format))
                decode.append(cl)
            self.decode = [decode, LayerConv('conv_post', [width, width], [n, colors], data_format=data_format)]
        self.variables = tf.contrib.framework.get_variables(vs)

    def __call__(self, data, carry, train=True):
        vout = data
        layers = []
        for x, convs in enumerate(self.decode[0]):
            vout = tf.concat([vout, data], 3) if x else vout
            vout = unboxn(convs[0](vout), 2)
            data = unboxn(data, 2)
            for conv in convs[1:]:
                vtmp = tf.nn.elu(conv(vout))
                vout = carry * vout + (1 - carry) * vtmp
            layers.append(vout)
        return self.decode[1](vout), self.variables
