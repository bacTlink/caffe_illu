#!/usr/bin/env python
# encoding: utf-8

################################################################
# File Name: build_resnet.py
# Author: gaoyu
# Mail: 1400012705@pku.edu.cn
# Created Time: 2018-02-08 16:17:34
################################################################

import argparse

from util import find_caffe
import caffe
from caffe import layers as L, params as P

def Build_Train_Data(prefix, batch_size):
    label_data_file = prefix + "/label,data"
    label_data = \
        L.Data(source = label_data_file,
               backend = P.Data.LMDB,
               batch_size = batch_size,
               transform_param = dict(
                   mirror = True,
                   crop_size = 224))
    label, data = L.Slice(label_data, slice_param = dict(slice_point = [3]), ntop = 2)
    return data, label

def Build_Test_Data(input_shape = [1, 18, 224, 224]):
    label_data = L.Input(input_param = dict(
        shape = dict(dim = input_shape)))
    label, data = L.Slice(label_data, slice_param = dict(slice_point = [3]), ntop = 2)
    return data, label

def Build_Data(split, prefix, batch_size):
    if split == "train":
        return Build_Train_Data(prefix, batch_size)
    elif split == "test":
        return Build_Test_Data()

def conv(bottom, channels, name = None):
    if (name is None):
        return L.Convolution(bottom,
                             num_output = channels,
                             kernel_size = 3,
                             stride = 1, pad = 1,
                             weight_filler = dict(type = 'msra'),
                             bias_filler = dict(type = 'constant'),
                             param = [dict(lr_mult = 1.0), dict(lr_mult = 0.1)])
    return L.Convolution(bottom,
                         name = name,
                         num_output = channels,
                         kernel_size = 3,
                         stride = 1, pad = 1,
                         weight_filler = dict(type = 'msra'),
                         bias_filler = dict(type = 'constant'),
                         param = [dict(lr_mult = 1.0), dict(lr_mult = 0.1)])

def bn(split, bottom):
    use_global_stats = (split == 'test')
    bn0 = L.BatchNorm(bottom,
                      param = [dict(lr_mult = 0, decay_mult = 0)] * 3,
                      batch_norm_param = dict(use_global_stats = use_global_stats))
    sc0 = L.Scale(bn0,
                  param = [dict(lr_mult = 1, decay_mult = 1)] * 2,
                  scale_param = dict(bias_term = True))
    return sc0

def eltwise(bottom1, bottom2):
    return L.Eltwise(bottom1, bottom2,
                     eltwise_param = dict(operation = P.Eltwise.SUM, coeff = [1, 1]))

def relu(bottom):
    return L.ReLU(bottom, in_place = True)

def ResNet_Block(split, bottom, channels, insert_bn):
    conv1 = conv(bottom, channels)
    if insert_bn:
        conv1 = bn(split, conv1)
    relu1 = relu(conv1)
    conv2 = conv(relu1, channels)
    if insert_bn:
        conv2 = bn(split, conv2)
    res = eltwise(bottom, conv2)
    return res

def Build_Resnet(split, bottom, resnet_blocks_num, channels, insert_bn):
    bottom = conv(bottom, channels)
    bottom = relu(bottom)
    last = bottom
    for i in xrange(resnet_blocks_num):
        last = ResNet_Block(split, last, channels, insert_bn)
    result = eltwise(bottom, last)
    result = conv(result, 3, "Output")
    return result

def Build_Loss(split, label, data, pic):
    average_data = L.Convolution(data, kernel_size = 1, stride = 1, num_output = 1,
            name = 'AverageData', 
            pad = 0, bias_term = False,
            weight_filler = dict(type = 'constant', value = 0.1),
            param = [dict(lr_mult = 0,decay_mult = 0)]
            )
    ref_loss = 0
    loss = L.EuclideanLoss(pic, label, propagate_down = [1, 0])
    ref_loss = L.EuclideanLoss(average_data, label, propagate_down = [0, 0], loss_weight = 0)
    return loss, ref_loss

def make_net(split, prefix, batch_size):
    net = caffe.NetSpec()
    data, label = Build_Data(split, prefix, batch_size)
    pic = Build_Resnet(split, data,
                       args.resnet_blocks_num,
                       args.channel,
                       args.insert_bn)
    (loss, ref_loss) = Build_Loss(split, label, data, pic)
    net.Loss = loss
    net.RefLoss = ref_loss
    return net.to_proto()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default = ".")
    p.add_argument("--batch", default = 20, type = int)
    p.add_argument("--resnet_blocks_num", default = 8, type = int)
    p.add_argument("--channel", default = 64, type = int, help = "convolution output_num")
    p.add_argument("--prefix", default = "/data3/lzh/1000x224x224", help = "data dir")
    p.add_argument("--insert_bn", default = False, action = "store_true", help = "insert batch_norm in resnet")
    args = p.parse_args()

    with open(args.output + "/train_resnet_auto.prototxt", "w") as f:
        f.write(str(make_net("train", args.prefix, args.batch)))
    with open(args.output + "/test_resnet_auto.prototxt", "w") as f:
        f.write(str(make_net("test", "", 1)))
