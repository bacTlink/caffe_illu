#!/usr/bin/env python
##########################################################
# File Name: conv_BN_scale_relu.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 17:05:43
##########################################################

from pycaffe import *

class ConvBNScaleReLU:
    def __init__(self, net = None, **kwargs):
        self.net = net or {}
        self._param = kwargs
        self._subp = None

    def getparam(self, key, default = None, **kwargs):
        if key in kwargs:
            return kwargs[key]
        if key in self.kwargs:
            return self.kwargs[key]
        if key in self._param:
            return self._param[key]
        try:
            return self.net._param["layer"][self._subp][key]
        except (TypeError, IndexError):
            pass
        if key in self.net._param:
            return self.net._param[key]
        if default is None:
            print "param %s not found" % key
            raise IndexError
        return default

    def __call__(self, *kargs, **kwargs):
        return self.conv_BN_scale_relu(*kargs, **kwargs)

    def conv_BN_scale_relu(self, bottom, 
                           bn = False,
                           relu = True,
                           **kargs):
        self.kwargs = kwargs
        top = L.Convolution(bottom, **self.conv_param())
        if bn:
            top = L.BatchNorm(top, **self.bn_param())
            top = L.scale(top, **self.scale_param())
        if relu:
            top = self.net.relu(top, in_place = True)
        return top
    
    def conv_param(self):
        self._subp = "conv"
        kernel = self.getparam("kernel")
        stride = self.getparam("stride", 1)
        pad = self.getparam("pad", (kernel - 1) / 2)
        param = {
            "kernel_size": kernel,
            "num_output" : self.getparam("channel"),
            "stride"     : stride,
            "pad"        : pad,
            "bias_term"  : self.getparam("bias_term", True),
            "weight_filler" : {
                "type": self.getparam("weight_filler", 'msra')},
            "bias_filler" : {
                "type": self.getparam("bias_filler", 'constant')},
            }
        if "name" in self.kwargs:
            param["name"] = self.kwargs["name"]
        self.check_freeze(param, 2) #conv has 2 blobs
        return param
    
    def bn_param(self):
        self._subp = "batchnorm"
        use_global_stats = self.net.use_global_stats
        param = {
            "batch_norm_param": {
                "use_global_stats": use_global_stats,
            },
            "in_place" = True,
            "param" = [dict(lr_mult = 0, decay_mult = 0)]*3,
            }
        if "name" in self.kwargs:
            param["name"] = self.kwargs["name"] + "_bn"
        return param

    def scale_param(self):
        self._subp = "scale"
        param = {
            "scale_param" : dict(bias_term = True),
            "in_place" : True,
            "param" : dict(lr_mult = self.getparam("lr_mult"),
            }
        if "name" in self.kwargs:
            param["name"] = self.kwargs["name"] + "_scale"
        self.check_freeze(param, 2) #scale has 2 blobs

    def check_freeze(self, param, blob_num):
        freeze = self.net.freeze
        if freeze:
            param["param"] = [dict(lr_mult = 0, decay_mult)]*blob_num
        return param
