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

def build_Data(net, split, prefix, batch_size, mode = "FLUX_DIS"):
    if split == "train":
        return build_Train_Data(net, prefix, batch_size, mode)
    elif split == "test":
        if mode == "FLUX_DIS":
            return build_Test_Data(net)
        elif mode == "PICS":
            return build_Test_Data_Pic(net)
        else:
            print "Unknown type"

def build_Train_Data(net, prefix, batch_size, mode = "FLUX_DIS"):
    if mode == "FLUX_DIS":
        dis_data_file = prefix + "/raw_data_photon_dis"
        flux_data_file = prefix + "/raw_data_photon_flux"
        dis_scale = 0.01000000
        flux_scale = 1.0

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
        data = L.Concat(dis_data, flux_data, ntop = 1)

        label_file = prefix + "/raw_data_conv"
        label_scale = 0.00390625

        label, label_name = \
            L.Data(source = label_file,
                   backend = P.Data.LMDB,
                   batch_size = batch_size,
                   ntop = 2,
                   transform_param = dict(scale = label_scale))
        if args.shuffle_channel:
            dis_data, flux_data = L.ShuffleChannel(dis_data, flux_data, ntop = 2);
        net.Silence = L.Silence(dis_name, flux_name, label_name,
                        ntop = 0)
    elif mode == "PICS":
        label_data_file = prefix + "/label,data"

        label_data = \
            L.Data(source = label_data_file,
                   backend = P.Data.LMDB,
                   batch_size = batch_size,
                   transform_param = dict(
                       mirror = True,
                       crop_size = 224))
        label, data = L.Slice(label_data, slice_param = dict(slice_point = [1]), ntop = 2)
        net.Silence = L.Silence(label_data,
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

def build_Test_Data_Pic(net,
                        input_shape = [1, 10, 224, 224]):
    data = L.Input(input_param = dict(
        shape = dict(dim = input_shape)))
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
    scale = L.Scale(BN,scale_param = dict(bias_term = True),in_place = True, param = dict(lr_mult = 0.01))
    #relu = L.ReLU(scale,in_place = True)
    relu = L.PReLU(scale,in_place = True)
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
    wise_relu = L.PReLU(wise,in_place = True)
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
            result = L.PReLU(result, in_place = True)
            snum -= 1
        else:
            result = relu
        nout /= 2
    #    if (s + 1) == stage_num:
    #        nout /= 2
    return result

def deriv_h(bottom, name, num):
    return L.Convolution(bottom,
            name = name,
            param = [dict(lr_mult = 0,decay_mult = 0)],
            num_output = num, group = num,
            stride = 1,
            kernel_h = 2, kernel_w = 1,
            pad_h = 1, pad_w = 0,
            bias_term = False,
            weight_filler = dict(type = 'bilateral', min = -1, max = 1),
            )

def deriv_w(bottom, name):
    return L.Convolution(bottom,
            name = name,
            param = [dict(lr_mult = 0,decay_mult = 0)],
            num_output = num, group = num,
            stride = 1,
            kernel_h = 1, kernel_w = 2,
            pad_h = 0, pad_w = 1,
            bias_term = False,
            weight_filler = dict(type = 'bilateral', min = -1, max = 1),
            )

def deriv(bottom, name, num):
    d_h = deriv_h(bottom, name + 'H', num)
    d_w = deriv_h(bottom, name + 'W', num)
    d_c_h = L.Crop(d_h, bottom, crop_param = dict(
        axis = 2,
        offset = [0, 0]
        ))
    d_c_w = L.Crop(d_w, bottom, crop_param = dict(
        axis = 2,
        offset = [0, 0]
        ))
    return L.Concat(d_c_h, d_c_w, name = name)

def build_Loss(split, data, bottom, label):
    net_result = L.Convolution(bottom,kernel_size = 1,stride = 1,num_output = 1,
                         pad = 0,bias_term = True,
                         weight_filler = dict(type = 'xavier'),
                         bias_filler = dict(type = 'constant'))
    net_result = L.PReLU(net_result, in_place = True)
    average_data = L.Convolution(data, kernel_size = 1, stride = 1, num_output = 1,
            name = 'AverageData', 
            pad = 0, bias_term = False,
            weight_filler = dict(type = 'constant', value = 0.1),
            param = [dict(lr_mult = 0,decay_mult = 0)]
            )
    pic = L.Eltwise(net_result,average_data,name='Sum',operation = P.Eltwise.SUM, propagate_down = [1, 0])
    ref_loss = 0
    if split == 'train':
        loss = L.EuclideanLoss(pic, label, propagate_down = [1, 0])
        ref_loss = L.EuclideanLoss(average_data, label, propagate_down = [0, 0], loss_weight = 0)
    else:
        loss = pic

    loss_deriv = 0
    ref_loss_deriv = 0
    if split == 'train' and (args.deriv or args.sec_deriv):
        label_deriv = deriv(label, 'LabelDeriv', 1)
        data_deriv = deriv(average_data, 'DataDeriv', 1)
        pic_deriv = deriv(pic, 'PicDeriv', 1)
        loss_deriv = L.EuclideanLoss(pic_deriv, label_deriv, propagate_down = [1, 0], loss_weight = 1)
        ref_loss_deriv = L.EuclideanLoss(data_deriv, label_deriv, propagate_down = [0, 0], loss_weight = 0)

    loss_sec_deriv = 0
    ref_loss_sec_deriv = 0
    if split == 'train' and args.sec_deriv:
        label_sec_deriv = deriv(label_deriv, 'LabelSecDeriv', 2)
        data_sec_deriv = deriv(data_deriv, 'DataSecDeriv', 2)
        pic_sec_deriv = deriv(pic_deriv, 'PicSecDeriv', 2)
        loss_sec_deriv = L.EuclideanLoss(pic_sec_deriv, label_sec_deriv, propagate_down = [1, 0], loss_weight = 0.5)
        ref_loss_sec_deriv = L.EuclideanLoss(data_sec_deriv, label_sec_deriv, propagate_down = [0, 0], loss_weight = 0)

    return loss, ref_loss, loss_deriv, ref_loss_deriv, loss_sec_deriv, ref_loss_sec_deriv

def make_net(split, prefix, batch_size, mode):
    global net
    net = caffe.NetSpec()
    data, label = build_Data(net, split, prefix, batch_size, mode)
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
    (loss, ref_loss,
            loss_deriv, ref_loss_deriv,
            loss_sec_deriv, ref_loss_sec_deriv
            ) = build_Loss(split, data, feature, label)
    if split == "train":
        net.Loss = loss
        net.RefLoss = ref_loss
        if args.deriv:
            net.LossDeriv = loss_deriv
            net.RefLossDeriv = ref_loss_deriv
        if args.sec_deriv:
            net.LossSecDeriv = loss_sec_deriv
            net.RefLossSecDeriv = ref_loss_sec_deriv
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
    p.add_argument("--deriv", default = False, action = "store_true", help = "use deriv loss")
    p.add_argument("--sec_deriv", default = False, action = "store_true", help = "use sec_deriv loss")
    args = p.parse_args()

    mode = "PICS"

    with open(args.output + "/train_resnet_auto.prototxt", "w") as f:
        f.write(str(make_net("train", args.prefix, args.batch, mode)))
    with open(args.output + "/test_resnet_auto.prototxt", "w") as f:
        f.write(str(make_net("test", "", 1, mode)))
