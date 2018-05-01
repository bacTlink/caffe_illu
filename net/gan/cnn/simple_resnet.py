#!/usr/bin/env python
##########################################################
# File Name: simple_resnet.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 18:39:08
##########################################################

from simple_cnn import *
from resnet_block import resnet_block as block

class SimpleResnet(SimpleCNN):
    def __init__(self, *kargs, **kwargs):
        SimpleCNN.__init__(self, *kargs, **kwargs)
        self._param["bottleneck"] = False
        self.brelu = lambda x: x
    
    def build(self, data):
        block_num = self._param["block_num"]
        top = conv(net = self)(data)
        first_conv = top
        #resnet block * n
        for i in xrange(block_num):
            top = block(top, kernel = 3, conv = self.conv, cnn_net = self)
        #output
        top = L.Eltwise(first_conv, top)
        top = self.conv(top, relu = False, 
                channel = 1, 
                name = "Output",
                **self.check_freeze(None, 2))
        return top
