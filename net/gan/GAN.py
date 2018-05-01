#!/usr/bin/env python
##########################################################
# File Name: GAN.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 23:12:56
##########################################################

import yaml
from pycaffe import *

import data_layer.illu_data
import cnn
import loss

class GAN:
    def __init__(self, args):
        self.param = yaml.load(open(args.net, 'r'))
        self.name = args.net.replace(".yaml", "")
        self.prefix = args.o

    def save_proto(self, net, token):
        fname = os.path.join(self.prefix, token + ".prototxt")
        with open(fname, 'w') as f:
            f.write(str(net.to_proto()))

    def make_data(self, mode):
        return data_layer.illu_data.illu_data(config = self.param, mode = mode)
    
    def make_cnn(self, session, freeze, mode):
        Type = self.param['net'][self.param['net'][session]]["type"]
        builder = getattr(cnn, Type)(session, self.param['net'], mode = mode)       
        builder.freeze = freeze
        builder.use_global_stats = (mode != "training") or (freeze) 
        return builder.build

    def make_loss(self, net, session, label, data, pic, cls):
        loss.GeneratorLoss(net, self.param).add_loss(label, data, pic)
        loss.ClassfierLoss(net, self.param).add_loss(cls, session)

    def deploy_net(self):       
        mode = "deploy"
        freeze = True
        net = caffe.NetSpec()
        data, label = self.make_data(mode)
        output = self.make_cnn("G", freeze, mode)(data)
        net.Output = output
        self.save_proto(net, "deploy")

    def concat_label_output(self, label, output):
        #build cls label with batch_size * 2
        return L.Concat(label, output,
                name = "ClsData",
                concat_param = dict(axis = 0),
                propagate_down  = [0, 1])

    def trainG_net(self):
        mode = "training"
        net = caffe.NetSpec()
        data, label = self.make_data(mode)
        output = self.make_cnn("G", False, mode)(data)
        net.Output = output
        concat = self.concat_label_output(label, output)
        cls = self.make_cnn("P", True, mode)(concat)
        net.Cls = cls
        self.make_loss(net, "G", label, data, output, cls)
        self.save_proto(net, "trainG")

    def trainP_net(self):
        mode = "training"
        net = caffe.NetSpec()
        data, label = self.make_data(mode)
        output = self.make_cnn("G", True, mode)(data)
        net.Output = output
        concat = self.concat_label_output(label, output)
        cls = self.make_cnn("P", False, mode)(concat)
        net.Cls = cls
        self.make_loss(net, "P", label, data, output, cls)
        self.save_proto(net, "trainP")

    def __call__(self):
        self.deploy_net()
        self.trainG_net()
        self.trainP_net()
