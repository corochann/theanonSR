""" Convolutional Neural Network
used for Super-Resolution Convolutional Neural Network (SRCNN)
"""
from __future__ import print_function

try:
    import cPickle as pickle
except:
    import pickle as pickle

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d


class ConvLayer(object):
    """ Convolution Layer
    """
    def __init__(self, rng, input, filter_shape, image_shape, activation=None,
                 W_values=None, b_values=None,
                 use_adam=False):
        """
        :param rng: RandomState - random number generator
        :param input: dtensor4 (batch_size, #of input feature map, image height, image width)
        :param filter_shape: (W is filter) tuple/list of length 4
        (#of filters (output feature map), #of input feature map, filter height, filter width)
        :param image_shape: tuple/list of length 4 (batch_size, #of input feature map, image height, image width)
        :param activation: Not used for now, instead we use LReLU as default
        :param W_values: None for initialize with random value (Training phase)
                  set numpy.ndarray for specifying W_value (Trained value will be set here in Application phase)
                  4D array - (#of filters (output feature map), #of input feature map, filter height, filter width)
                  [NOTE] self.W is theano.tensor.shared.var.TensorSharedVariable, but this W is numpy.adarray (only value)
        :param b_values: None for initialize with random value (Training phase)
                  set numpy.ndarray for specifying W_value (Trained value will be set here in Application phase)
                  [NOTE] self.b is theano.tensor.shared.var.TensorSharedVariable, but this b is numpy.adarray (only value)
                  1D array - (#of filters (output feature map), )
        :param use_adam: deprecated, should be always True. (True when use ADAM)
        :return:
        """
        assert image_shape[1] == filter_shape[1]  # make sure they are the same
        # pad outside, to maintain input image & output image's height/width same.
        self.input = input

        "fan_in - #of input to convolution layer = #of input feature maps * filter height * filter width"
        fan_in = np.prod(filter_shape[1:])
        "fan_out - #of output which one element affects to = #of output * filter height * filter width"
        fan_out = (filter_shape[0]) * np.prod(filter_shape[2:])

        # init @: 4d tensor(filter_shape)
        if W_values is None:
            # Training phase, init with random value
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            # W_bound = 1. / fan_in
            W_values = np.asarray(
                rng.uniform(low=-1.0*W_bound, high=1.2*W_bound, size=filter_shape),
                dtype=theano.config.floatX
            )

        self.W = theano.shared(W_values, borrow=True)

        # init bias: 1D tensor (#of output feature) - one bias per output feature map
        if b_values is None:
            # Training phase, init with 0
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        if use_adam:
            zero_W_values = np.zeros(filter_shape)
            self.W_m = theano.shared(value=zero_W_values, borrow=False)
            self.W_v = theano.shared(value=zero_W_values, borrow=False)
            self.b_m = theano.shared(value=b_values, borrow=False)
            self.b_v = theano.shared(value=b_values, borrow=False)  # for momentum

        # CONVOLUTION
        # image_shape: tuple/list of length 4 (batch_size, #of input feature map, image height, image width)
        "conv_out: 4d tensor (batch_size, #of output feature map, output height, output width)"
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            border_mode='valid',
            input_shape=image_shape
        )

        conv_out_plus_b = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        # TODO: consider activation
        # - sigmoid, tanh, LeRU (max(0, x)) etc.

        # - relu: vanishing gradient occurs
        # self.output = T.minimum(T.maximum(0, conv_out + self.b.dimshuffle('x', 0, 'x', 'x')), 1)

        # - lrelu: leak rectified linear unit
        # self.output = T.switch(T.lt(conv_out_plus_b, 0), -0.1*conv_out_plus_b, conv_out_plus_b) # LReLU

        # - cropped leaky relu -> cropped to 0-1
        self.output = T.switch(T.lt(conv_out_plus_b, - 1179 / 256.), 0,
                               T.switch(T.lt(conv_out_plus_b, 1 / 256.), conv_out_plus_b / 1280. + 1279 / 327680.,
                                        T.switch(T.lt(conv_out_plus_b, 255 / 256.), conv_out_plus_b,
                                                 T.switch(T.lt(conv_out_plus_b, 1535 / 256.), conv_out_plus_b / 1280. + 326145. / 327680.,
                                                          1))))
        self.params = [self.W, self.b]

        if use_adam:
            self.params_m = [self.W_m, self.b_m]
            self.params_v = [self.W_v, self.b_v]

    def cost(self, y):
        """ calculating cost by Mean Squared Error (MSE) as the loss function
        NOTE: y and self.output must be same tensor size
        :param y: 4d tensor (batch_size, #of feature map (3 for RGB), output height, output width)
        :return:
        """
        return T.mean(T.sqr(y - self.output))
