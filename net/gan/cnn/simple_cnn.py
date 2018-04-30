#!/usr/bin/env python
##########################################################
# File Name: simple_cnn.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 21:29:16
##########################################################

from pycaffe import L
from conv_BN_scale_relu import conv_BN_scale_relu as conv

class SimpleCNN:
    def __init__(self, Type, config = None, mode = None, **kwargs):
        self._param = {}
        self.name = config[Type]
        self._param.update(config)
        self._param.update(config[self.name])
        self.relu = L.ReLU
        self._param["mode"] = mode or "training"

    def build(self, data):
        conv_num = self._param["conv_num"]
        top = data
        stages = []
        conv_bn_scale_relu = conv(net = self)
        top = conv_bn_scale_relu(top, 
                bn = True,
                relu = True,
                kernel = 7,
                stride = 2,
                pad = 3)
        top = L.Pooling(top, 
                pooling_param = dict(kernel_size = 3,
                                     stride = 2,
                                     pool = 0))
        channel = self._param["channel"]
        channel_plus = self._param["channel_plus"]
        for i in xrange(conv_num):
            channel *= channel_plus
            top = conv_bn_scale_relu(top, 
                    bn = True, 
                    relu = True, 
                    channel = channel)
        top = L.Pooling(top,
                pooling_param = dict(
                    kernel_size = self._param["map_size"],
                    stride = 1,
                    pool = 1))
        top = L.InnerProduct(top,
                name = "fc",
                inner_product_param = dict(num_output = 2))
        return top
