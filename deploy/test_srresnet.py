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

#caffe_path = "/home/gaoyu/caffe_illu/python/"
caffe_path = "/home/linzehui/illu/caffe_illu/python/"
sys.path.append(caffe_path)

import numpy as np
import caffe
import lmdb

import cv2

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
    return net


def GetOnePic(net, index, use_train = False):
    index *= 3
    if use_train:
        #prefix = "/data3/lzh/10000x10x224x224_ring_colored_diff_filtered/"
        prefix = "/data3/lzh/10000x10x224x224_box_diff/train-"
        #prefix = "/data3/lzh/10000x10x224x224_Diamond_diff/train-"
    else:
        #prefix = "/data3/lzh/10000x10x224x224_ring_colored_diff_filtered/"
        prefix = "/data3/lzh/10000x10x224x224_box_diff/test-"
        #prefix = "/data3/lzh/10000x10x224x224_Diamond_diff/test-"
    for i in xrange(3):
        if (i == 0):
            tmpdata = GetOneFromLMDB(prefix + "label,data/", index + i)
            label = tmpdata[0, :, :].reshape(1, 224, 224)
            data = tmpdata.reshape(1, tmpdata.shape[0], tmpdata.shape[1], tmpdata.shape[2])
        else:
            tmpdata = GetOneFromLMDB(prefix + "label,data/", index + i)
            data = np.append(data, tmpdata.reshape(1, tmpdata.shape[0], tmpdata.shape[1], tmpdata.shape[2]), axis = 0)
            label = np.append(tmpdata[0, :, :].reshape(1, 224, 224), label, axis = 0)
        net.blobs['Input1'].data[...] = data[i]
        net.forward()
        if i == 0:
            output = net.blobs['Convolution18'].data[0]
        else:
            output = np.append(net.blobs['Convolution18'].data[0], output, axis = 0)
    index /= 3

    output = np.maximum(0, output)
    output = np.minimum(1, output)
    label = np.maximum(0, label)
    label = np.minimum(1, label)
    print output.shape
    pic = (output * 255).copy()
    pic = pic.astype(np.uint8)
    pic = np.transpose(pic, (1, 2, 0))
    cv2.imwrite("test" + str(index) + ".png", pic)
    pic2 = (label * 255).copy()
    pic2 = np.transpose(pic2, (1, 2, 0))
    cv2.imwrite("train" + str(index) + ".png", pic2)
    loss = 0
    for i in range(224):
        for j in range(224):
            v = float(pic[i][j][0] - pic2[i][j][0]) * (1.0 / 255.0)
            loss += v * v
    print loss;

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

