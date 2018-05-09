#!/usr/bin/env python
# encoding: utf-8

################################################################
# File Name: test_net.py
# Author: gaoyu
# Mail: 1400012705@pku.edu.cn
# Created Time: 2018-03-04 15:37:26
################################################################

import os, sys
import argparse
caffe_path = "/home/linzehui/illu/caffe_illu/python/"
sys.path.append(caffe_path)
import numpy as np
import caffe
import lmdb
import skimage.io

def GetOneFromLMDB(lmdb_path, index = 0):
    lmdb_env = lmdb.open(lmdb_path, readonly = True)
    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        i = -1
        for key, value in cursor:
            raw_datum = value
            i += 1
            #the 1st pic
            if i == index:
                break

    datum = caffe.proto.caffe_pb2.Datum()
    datum.ParseFromString(raw_datum)

    flat_x = np.fromstring(datum.data, dtype = np.uint8)
    is_int = flat_x.shape[0] > 0

    if not is_int:
        #float_data
        x = [v for v in datum.float_data]
        x = np.array(x)
        x = x.reshape(datum.channels, datum.height, datum.width)
    else:
        #int_data
        x = flat_x.reshape(datum.channels, datum.height, datum.width)
    return x

def GetNet(net, weights, gpu):
    caffe.set_device(gpu)
    caffe.set_mode_gpu()
    net = caffe.Net(net, weights, caffe.TEST)
    #net = caffe.Net(net, caffe.TEST)
    return net


def GetOnePic(net, index, use_train = False):
    if use_train:
        prefix = "/data3/lzh/10000x10x224x224_box_colored_diff/train-"
        #prefix = "/data3/lzh/10000x10x224x224_Diamond_colored_diff/train-"
    else:
        prefix = "/data3/lzh/10000x10x224x224_box_colored_diff/test-"
        #prefix = "/data3/lzh/10000x10x224x224_Diamond_colored_diff/test-"
    data = GetOneFromLMDB(prefix + "label,data/", index)
    net.blobs['Input1'].data[...] = data
    net.forward()
    output = net.blobs['Convolution22'].data[0]
    label = net.blobs['Slice1'].data[0]
    loss = net.blobs['Loss'].data
    print loss

    output = np.maximum(0, output)
    output = np.minimum(1, output)
    print output.shape
    pic = np.transpose(output, (1, 2, 0))
    pic2 = np.transpose(label, (1, 2, 0))
    res = np.concatenate((pic, pic2), axis = 1)
    skimage.io.imsave("res" + str(index) + ".png", res)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type = int, default = 0)
    p.add_argument("--net", default = "/home/gaoyu/illu_train/test_resnet_auto.prototxt")
    p.add_argument("--train", action = "store_true", default = False)
    p.add_argument("model")

    args = p.parse_args()
    net = GetNet(args.net,
                 args.model,
                 gpu = args.gpu)
    for i in range(10):
        print "INDEX: ", i
        print "------------------------------------"
        GetOnePic(net, i, args.train)

