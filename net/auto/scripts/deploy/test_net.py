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

caffe_path = "/home/gaoyu/caffe_illu/python/"
sys.path.append(caffe_path)

import numpy as np
import caffe
import lmdb

import cv2

def GetOneFromLMDB(lmdb_path, index = 1):
    lmdb_env = lmdb.open(lmdb_path, readonly = True)
    with lmdb_env.begin() as lmdb_txn:
        cursor = lmdb_txn.cursor()
        i = 0
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
    if use_train:
        prefix = "/data3/lzh/1000x224x224/"
    else:
        prefix = "/data3/lzh/100x224x224/"
    dis_data = GetOneFromLMDB( \
        prefix + "raw_data_photon_dis/", index)
    flux_data = GetOneFromLMDB( \
        prefix + "raw_data_photon_flux/", index)
    label = GetOneFromLMDB( \
        prefix + "raw_data_conv/", index)

    net.blobs['Data1'].data[...] = dis_data *  0.01
    net.blobs['Data3'].data[...] = flux_data

    net.forward()

    output = net.blobs['Convolution22'].data[0]
    print output * 256
    print label
    output = np.maximum(0, output)
    output = np.minimum(0.9999999, output)
    pic = np.zeros(output.shape, np.uint8)
    pic = (output * 256).copy()
    pic = pic.astype(np.uint8)
    pic = np.transpose(pic, (1, 2, 0))
    cv2.imwrite("test" + str(index) + ".jpg", pic)
    pic2 = np.zeros(label.shape, np.uint8)
    pic2 = label.copy()
    pic2 = np.transpose(pic2, (1, 2, 0))
    cv2.imwrite("train" + str(index) + ".jpg", pic2)
    loss = 0
    for i in range(224):
        for j in range(224):
            v = float(int(pic[i][j][0]) - int(pic2[i][j][0])) * (1.0 / 256.0)
            loss += v * v
    print loss

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
        GetOnePic(net, i + 1, args.train)

