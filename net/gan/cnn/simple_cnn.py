#!/usr/bin/env python
##########################################################
# File Name: simple_cnn.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 21:29:16
##########################################################

from pycaffe import L
from conv_BN_scale_relu import ConvBNScaleReLU as conv

class SimpleCNN:
    def __init__(self, session, config = None, mode = None, **kwargs):
        self._param = {}
        self.name = config[session]
        self._param.update(config)
        self._param.update(config[self.name])
        self.relu = L.ReLU
        self._param["mode"] = mode or "training"
        self.conv = conv(self)
        self.layer_cnt = {}

    def layer_name(self, token):
        if not token in self.layer_cnt:
            self.layer_cnt[token] = 0
        self.layer_cnt[token] += 1
        return "%s_%s%d" % (self.name, token, self.layer_cnt[token])

    def build(self, data):
        conv_num = self._param["conv_num"]
        top = data
        stages = []
        top = self.conv(top, 
                name = self.layer_name("convolution"),
                bn = True,
                relu = True,
                kernel = 7,
                stride = 2,
                pad = 3)
        top = L.Pooling(top, 
                name = self.layer_name("pooling"),
                pooling_param = dict(kernel_size = 3,
                                     stride = 2,
                                     pool = 0))
        channel = self._param["channel"]
        channel_plus = self._param["channel_plus"]
        for i in xrange(conv_num):
            channel *= channel_plus
            top = self.conv(top, 
                    bn = True, 
                    relu = True, 
                    channel = channel)
        top = L.Pooling(top,
                name = self.layer_name("pooling"),
                pooling_param = dict(
                    kernel_size = self._param["map_size"],
                    stride = 1,
                    pool = 1))
        top = L.InnerProduct(top,
                name = "fc",
                inner_product_param = dict(num_output = 2),
                **self.check_freeze(None, 2))
        return top

    def check_freeze(self, param, blob_num):
        param = param or dict()
        freeze = self.freeze
        if freeze:
            param["param"] = [dict(lr_mult = 0, decay_mult = 0)]*blob_num
        return param
