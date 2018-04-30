#!/usr/bin/env python
##########################################################
# File Name: resnet_block.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 18:14:08
##########################################################

from pycaffe import L

def resnet_block(bottom, conv, net = None, **kwargs):
    try:
        bottleneck = net._param["bottleneck"]
    except:
        bottleneck = False  

    if args.bottleneck:
        channel = kwargs.pop("channel")
        kernel = kwargs.pop("kernel") if "kernel" in kwargs else 3
        top = conv(kernel = 1, channel = channel / 4, **kwargs) 
        top = conv(kernel = 3, channel = channel / 4, **kwargs)
        top = conv(kernel = 1, channel = channel, relu = False, **kwargs)
    else:
        top = conv(**kwargs)
        top = conv(**kwargs, relu = False)
    top = L.Eltwise(bottom, top)
    top = self.net.brelu(top)
    return top
