#!/usr/bin/env python
##########################################################
# File Name: pycaffe.py
# Author: gaoyu
# mail: gaoyu14@pku.edu.cn
# Created Time: 2018-04-30 18:56:33
##########################################################

import os, sys

caffe_path = "/home/gaoyu/caffe_illu/python"
if os.path.exists(caffe_path):
    sys.path.append(caffe_path)
else:
    print "caffe not found"
    sys.exit(2)

import caffe
from caffe import layers as L
