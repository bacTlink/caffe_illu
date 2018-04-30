#!/usr/bin/env python
##########################################################
# File Name: generator.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 19:50:25
##########################################################

from pycaffe import L

class GeneratorLoss(BaseLoss):
    def add_loss(self, label, data, pic):
        net = self.net
        if self.getparam("RefLoss"):
            average_data = L.Convolution(data, 
                kernel_size = 1, 
                stride = 1, 
                num_output = 1,
                name = 'AverageData', 
                pad = 0, bias_term = False,
                weight_filler = dict(type='constant',value = 0.1),
                param = [dict(lr_mult = 0,decay_mult = 0)]
                )
            ref_loss = L.EuclideanLoss(average_data, label, propagate_down = [0, 0], loss_weight = 0)
            net.RefLoss = ref_loss
        if self.getparam("MSELoss"):
            loss = L.EuclideanLoss(pic, label, propagate_down = [0, 0], loss_weight = 0)
            net.GeneratorLoss = loss
