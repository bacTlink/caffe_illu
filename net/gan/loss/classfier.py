#!/usr/bin/env python
##########################################################
# File Name: classfier.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 20:21:33
##########################################################

from base_loss import *
from pycaffe import L

class ClassfierLoss(BaseLoss):
    def GAN_label(self):
        assert(hasattr(self, "session"))
        batch_size = self._param["data"]["batch_size"]

        dummydata = lambda name, value: \
                L.DummyData(
                        name = name,
                        dummy_data_param = dict(
                            shape = dict(dim = [batch_size,]),
                            data_filler = dict(value = value)))
        
        neg_label = dummydata("NegLabel", 0)
        pos_label = dummydata("PosLabel", 1)
        
        #concat bottom 0 as gt, bottom 1 as pr
        #when training generator, gt should be false, pr should be true
        #when training classfier, gt should be true, pr should be false
        if self.session == "G":
            return L.Concat(neg_label, pos_label, 
                    name = "OppSoftmaxLabel",
                    axis = 0)
        elif self.session == "P":
            return L.Concat(pos_label, neg_label,
                    name = "SoftmaxLabel",
                    axis = 0)
        else:
            raise Exception("LossError", "Unknown session") 

    def add_loss(self, classfier, session = None):
        self.session = session or self.session
        softmax_label = self.GAN_label()
        loss = L.SoftmaxWithLoss(classfier, softmax_label, 
                loss_weight = 1000) 
        accuracy = L.Accuracy(classfier, softmax_label)
        self.net.ClassfierLoss = loss
        self.net.Accuraccy = accuracy 
        return loss
