#!/usr/bin/env python
##########################################################
# File Name: GAN.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 23:12:56
##########################################################

import yaml
import pycaffe

import data_layer
import cnn
import loss

class GAN:
    def __init__(self, args):
        self.param = yaml.load(args.net)
        self.name = args.net.replace(".yaml", "")
        self.prefix = args.o
        self.GNet = self.param["net"]["G"]
        self.PNet = self.param["net"]["P"]

    def save_proto(self, net, token):
        fname = os.path.join(self.prefix, token + ".prototxt")
        with open(fname, 'w') as f:
            f.write(str(net.to_proto()))

    def make_data(self, mode):
        return data_layer.illu_data.illu_data(config = self.param, mode = mode)
    
    def make_cnn(self, Type, freeze, mode):
        builder = getattr(cnn, Type)(Type, self.param['net'], mode = mode)       
        builder.freeze = freeze
        builder.use_global_stats = (mode != "training") or (freeze) 
        return builder.build

    def make_loss(self, net, session, label, data, pic, cls):
        loss.GeneratoraLoss(net, self.param).add_loss(label, data, pic)
        loss.ClassfierLoss(net, self.param).add_loss(cls, session)

    def deploy_net(self):       
        mode = "deploy"
        freeze = True
        net = caffe.net.NetSpec("deploy")
        data, label = self.make_data(mode)
        output = self.make_cnn(self.GNet, "G", freeze, mode)(data)
        net.Output = output
        self.save_proto(net, "deploy")

    def trainG_net(self):
        mode = "training"
        net = caffe.net.NetSpec("trainG")
        data, label = self.make_data(mode)
        output = self.make_cnn(self.GNet, "G", False, mode)(data)
        #build cls label with batch_size * 2
        concat = L.pycaffe.Concat(label, output, 
                name = "ClsData",
                concat_param = dict(axis = 0))
        cls = self.make_cnn(self.PNet, "P", True, mode)(data)
        self.make_loss(net, "G", label, data, output, cls)
        net.Output = output
        net.Cls = cls
        self.save_proto(net, "trainG")

    def trainP_net(self):
        mode = "training"
        net = caffe.net.NetSpec("trainP")
        data, label = self.make_data(mode)
        output = self.make_cnn(self.GNet, "G", True, mode)(data)
        #build cls label with batch_size * 2
        concat = L.pycaffe.Concat(label, output, 
                name = "ClsData",
                concat_param = dict(axis = 0))
        cls = self.make_cnn(self.PNet, "P", False, mode)(data)
        self.make_loss(net, "P", label, data, output, cls)
        net.Output = output
        net.Cls = cls
        self.save_proto(net, "trainP")

    def __call__(self):
        self.deploy_net()
        self.trainG_net()
        self.trainP_net()
