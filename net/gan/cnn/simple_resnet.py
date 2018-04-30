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
        top = data
        stages = []
        #resnet block * n
        for i in xrange(block_num):
            top = block(top, kernel = 3, conv = conv, net = self)
            stages.append(top)
        #output
        top = L.Eltwise(stages[0], top)
        top = conv(top, relu = False, name = "Output")
        return top
