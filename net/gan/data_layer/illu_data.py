#!/usr/bin/env python
##########################################################
# File Name: illu_data.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 18:53:00
##########################################################

from pycaffe import L, caffe

def Build_Train_Data(prefix, batch_size):
    label_data_file = prefix + "label,data"
    label_data = \
        L.Data(source = label_data_file,
               backend = caffe.params.Data.LMDB,
               batch_size = batch_size,
               transform_param = dict(
                   mirror = True,
                   crop_size = 224))
    label, data = L.Slice(label_data, slice_param = dict(slice_point = [1]), ntop = 2)
    return data, label

def Build_Test_Data(input_shape = [1, 11, 224, 224]):
    label_data = L.Input(name = "Input",
        input_param = dict(shape = dict(dim = input_shape)))
    label, data = L.Slice(label_data, slice_param = dict(slice_point = [1]), ntop = 2)
    return data, label

def Build_Data(split, prefix, batch_size):
    if split == "train":
        return Build_Train_Data(prefix, batch_size)
    elif split == "test":
        return Build_Test_Data()

def illu_data(mode, config):
    prefix = config["data"]["prefix"]
    batch_size = config["data"]["batch_size"]
    return Build_Data(split = "train" if mode == "training" else "test",
                      prefix = prefix,  
                      batch_size = batch_size)
