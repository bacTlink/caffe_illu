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
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import GAN

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("-o", default = None)
    p.add_argument("net", help = "net config")
    return p

def make_solver(session):
    res = ''
    res += 'net: "train%s.prototxt"\n' % session
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
    res += 'snapshot_prefix: "snapshots/gan%s/gan%s"\n' % (session, session)
    res += 'solver_mode: GPU\n'
    with open(args.o + "/solver%s.prototxt" % session, 'w') as f:
        f.write(res)

if __name__ == "__main__":
    args = parse().parse_args()
    if not args.o:
        args.o = os.path.basename(args.net).replace(".yaml", "")
        args.o = os.path.join(os.path.dirname(__file__) + "/prototxt/", args.o)
        if not os.path.isdir(args.o):
            os.mkdir(args.o)
    GAN.GAN(args)()
    map(make_solver, ["G", "P"])
