#!/usr/bin/env python
# encoding: utf-8

################################################################
# File Name: build.py
# Author: gaoyu
# Mail: 1400012705@pku.edu.cn
# Created Time: 2018-02-08 16:17:34
################################################################

import argparse

import os, sys
sys.path.append(os.path.realpath(__file__))

import GAN

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-o", default = ".")
    p.add_argument("net", help = "net config")
    args = p.parse_args()

def make_solver():
    res = ''
    res += 'net: "train_srresnet_auto.prototxt"\n'
    res += 'type: "Adam"\n'
    res += '# default momentum: 0.9\n'
    res += 'momentum: 0.9\n'
    res += '# default momentum2: 0.999\n'
    res += 'momentum2: 0.999\n'
    res += '# default delta: 1e-8\n'
    res += '# delta: 0.00000001\n'
    res += 'base_lr: 0.0001\n'
    res += 'lr_policy: "fixed"\n'
    res += 'display: 100\n'
    res += 'max_iter: 100000\n'
    res += 'snapshot: 10000\n'
    res += 'snapshot_prefix: "snapshots/gan"\n'
    res += 'solver_mode: GPU\n'
    return res;


if __name__ == "__main__":
    args = parse().parse_args()
    GAN.GAN(args)()
