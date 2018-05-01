#!/usr/bin/env python
##########################################################
# File Name: loss/base_loss.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 20:07:46
##########################################################

class BaseLoss:
    def __init__(self, net, config = None, **kwargs):
        self._param = {}
        self.net = net
        if config:
            self._param.update(config)
            self._param.update(config["loss"])

    def getparam(self, key, default = None):
        try:
            return self._param[key]
        except (TypeError, IndexError):
            return default
