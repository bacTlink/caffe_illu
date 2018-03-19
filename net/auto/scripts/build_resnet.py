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

def build_Data(net, split, prefix, batch_size):
    if split == "train":
        return build_Train_Data(net, prefix, batch_size)
    elif split == "test":
        return build_Test_Data(net)

def build_Train_Data(net, prefix, batch_size):
    dis_data_file = prefix + "/raw_data_photon_dis"
    flux_data_file = prefix + "/raw_data_photon_flux"
    dis_scale = 0.01000000
    flux_scale = 1.0

    label_file = prefix + "/raw_data_conv"
    label_scale = 0.00390625

    dis_data, dis_name = \
        L.Data(source = dis_data_file,
               backend = P.Data.LMDB,
               batch_size = batch_size,
               ntop = 2,
               transform_param = dict(scale = dis_scale))

    flux_data, flux_name = \
        L.Data(source = flux_data_file,
               backend = P.Data.LMDB,
               batch_size = batch_size,
               ntop = 2,
               transform_param = dict(scale = flux_scale))

    label, label_name = \
        L.Data(source = label_file,
               backend = P.Data.LMDB,
               batch_size = batch_size,
               ntop = 2,
               transform_param = dict(scale = label_scale))

    if args.shuffle_channel:
        dis_data, flux_data = L.ShuffleChannel(dis_data, flux_data, ntop = 2);

    data = L.Concat(dis_data, flux_data, ntop = 1)
    net.Silence = L.Silence(dis_name, flux_name, label_name,
                        ntop = 0)
    return data, label

def build_Test_Data(net,
                    dis_shape = [1, 20, 224, 224],
                    flux_shape = [1, 20, 224, 224]):
    dis_data = L.Input(input_param = dict(
        shape = dict(dim = dis_shape)))
    flux_data = L.Input(input_param = dict(
        shape = dict(dim = flux_shape)))
    net.Data1 = dis_data
    net.Data3 = flux_data

    data = L.Concat(dis_data, flux_data, ntop = 1)
    return data, 0

def build_Resnet(split, bottom, repeat, stage_num, channels, stage_list = None):
    #TODO: add pooling
    scale, result = conv_BN_scale_relu(split,
                           bottom,
                           nout = channels,
                           ks = 7,
                           stride = 2,
                           pad = 3)

 #   result = L.Pooling(result,
 #                      pooling_param = dict(kernel_size = 3,
 #                                           stride = 2,
 #                                           pool = 0)
 #                      )

    #channels *= 4
    if stage_list is None:
        stage_list = []
    for s in range(stage_num):
        for i in range(repeat):
            if i == 0:
                if s == 0:
                    projection_stride = 1
                else:
                    projection_stride = 2
            else:
                projection_stride = 1
            result = ResNet_block(split, result, nout = channels,
                                  projection_stride = projection_stride)
        stage_list.append(result)
        channels *= 2
    return result

def conv_BN_scale_relu(split,bottom,nout,ks,stride,pad=0, conv_type = "Convolution", name = None):
    if conv_type == "Convolution":
        conv = L.Convolution(bottom,kernel_size = ks,stride = stride,num_output = nout,
                         pad = pad,bias_term = True,
                         weight_filler = dict(type = 'xavier'),
                         bias_filler = dict(type = 'constant'))
    else:
        conv = L.Deconvolution(bottom,
                    convolution_param =
                    dict(kernel_size = ks,
                         stride = stride,
                         num_output = nout,
                         pad = pad,
                         bias_term = True,
                         weight_filler = dict(type = 'xavier'),
                         bias_filler = dict(type = 'constant')))
    if split == "train":
        use_global_stats = False
    else:
        use_global_stats = True
    BN = L.BatchNorm(conv,batch_norm_param = dict(use_global_stats = use_global_stats),
                     in_place = True,
                     param = [dict(lr_mult = 0,decay_mult = 0),
                              dict(lr_mult = 0,decay_mult = 0),
                              dict(lr_mult = 0,decay_mult = 0)])
    scale = L.Scale(BN,scale_param = dict(bias_term = True),in_place = True)
    relu = L.ReLU(scale,in_place = True)
    if not name is None:
        net[name] = conv
        net[name + "_BN"] = BN
        net[name + "_Scale"] = scale
        #net[name + "_ReLU"] = relu
    return scale, relu

def ResNet_block(split,bottom,nout,projection_stride):
    if projection_stride == 1:
        scale0 = bottom
    elif projection_stride > 1:
        scale0,relu0 = conv_BN_scale_relu(split,bottom,nout,1,projection_stride,0)
    elif projection_stride == 0:
        projection_stride = 1
        scale0,relu0 = conv_BN_scale_relu(split, bottom, nout, 1, 1, 0)

    if not args.bottleneck:
        #Resnet32
        scale1,relu1 = conv_BN_scale_relu(split,bottom,nout,3,projection_stride,1)
        scale2,relu2 = conv_BN_scale_relu(split,relu1,nout,3,1,1)
    else:
        #Resnet50
        scale1,relu1 = conv_BN_scale_relu(split,bottom,nout/4,1,projection_stride,0)
        scale3,relu3 = conv_BN_scale_relu(split,relu1,nout/4,3,1,1)
        scale2,relu2 = conv_BN_scale_relu(split,relu3,nout,1,1,0)

    wise = L.Eltwise(scale2,scale0,operation = P.Eltwise.SUM)
    wise_relu = L.ReLU(wise,in_place = True)
    return wise_relu

def build_UpSample(split, bottom, stage_num, channels, stage_list = []):
    result = bottom
    nout = channels / 2
    snum = len(stage_list) - 2
    for s in range(stage_num):
        scale, relu = conv_BN_scale_relu(split,result,nout,ks=2,stride=2,conv_type="Deconvolution")
        #      scale, result = conv_BN_scale_relu(split,result,nout,ks=1,stride=1)
        if snum >= 0:
            scale2, relu2 = conv_BN_scale_relu(split,stage_list[snum],nout,ks=1,stride=1,pad=0,
                                                     name = "StageLink" + str(snum))
            result = L.Eltwise(scale, scale2, operation = P.Eltwise.SUM)
            result = L.ReLU(result, in_place = True)
            snum -= 1
        else:
            result = relu
        nout /= 2
    #    if (s + 1) == stage_num:
    #        nout /= 2
    return result

def build_Loss(split, bottom, label):
    pic = L.Convolution(bottom,kernel_size = 1,stride = 1,num_output = 1,
                         pad = 0,bias_term = True,
                         weight_filler = dict(type = 'xavier'),
                         bias_filler = dict(type = 'constant'))
    pic = L.ReLU(pic, in_place = True)
    if split == "train":
        loss = L.EuclideanLoss(pic, label, propagate_down = [1, 0])
    else:
        loss = pic #TODO write to file
    return loss

def make_net(split, prefix, batch_size):
    global net
    net = caffe.NetSpec()
    data, label = build_Data(net, split, prefix, batch_size)
    basechannel = args.channel
    stage_num = args.stage_num
    stage_list = []
    feature = build_Resnet(split, data,
                           repeat = args.repeat,
                           stage_num = stage_num,
                           channels = basechannel,
                           stage_list = stage_list)
    if not args.stage_link:
        stage_list = []
    feature = build_UpSample(split, feature,
                             stage_num = stage_num,
                             channels = 256,
                             stage_list = stage_list)
    loss = build_Loss(split, feature, label)
    if split == "train":
        net.Loss = loss
    else:
        net.Output = loss
    return net.to_proto()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default = ".")
    p.add_argument("--batch", default = 24, type = int)
    p.add_argument("--stage_num", default = 3, type = int)
    p.add_argument("--repeat", default = 3, type = int, help = "the size of each resnet stage")
    p.add_argument("--channel", default = 32, type = int, help = "first convolution output_num")
    p.add_argument("--prefix", default = "/data3/lzh/1000x224x224", help = "data dir")
    p.add_argument("--stage_link", default = False, action = "store_true", help = "link resnet stage to Deconvolution")
    p.add_argument("--shuffle_channel", default = False, action = "store_true", help = "add shuffle channel layer to data")
    p.add_argument("--bottleneck", default = False, action = "store_true", help = "use bottleneck in resnet50")
    args = p.parse_args()

    with open(args.output + "/train_resnet_auto.prototxt", "w") as f:
        f.write(str(make_net("train", args.prefix, args.batch)))
    with open(args.output + "/test_resnet_auto.prototxt", "w") as f:
        f.write(str(make_net("test", "", 1)))
