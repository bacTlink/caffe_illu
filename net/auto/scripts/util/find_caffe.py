#!/usr/bin/env python
# encoding: utf-8

################################################################
# File Name: find_caffe.py
# Author: gaoyu
# Mail: 1400012705@pku.edu.cn
# Created Time: 2018-03-07 11:35:39
################################################################

import os, sys

caffe_path = "/home/gaoyu/caffe_illu/python"
if os.path.exists(caffe_path):
    sys.path.append(caffe_path)
else:
    print "caffe not found"
    sys.exit(2)
