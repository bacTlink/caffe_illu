#!/usr/bin/env python
##########################################################
# File Name: resnet_block.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 18:14:08
##########################################################

from pycaffe import L

def resnet_block(bottom, conv, cnn_net, **kwargs):
    try:
        bottleneck = cnn_net._param["bottleneck"]
    except:
        bottleneck = False  

    if bottleneck:
        channel = kwargs.pop("channel")
        kernel = kwargs.pop("kernel") if "kernel" in kwargs else 3
        top = conv(bottom, kernel = 1, channel = channel / 4, **kwargs) 
        top = conv(top, kernel = 3, channel = channel / 4, **kwargs)
        top = conv(top, kernel = 1, channel = channel, relu = False, **kwargs)
    else:
        top = conv(bottom, **kwargs)
        top = conv(top, relu = False, **kwargs)
    top = L.Eltwise(bottom, top)
    top = cnn_net.brelu(top)
    return top
