#!/usr/bin/env sh
set -e

CBIN=/home/linzehui/illu/build/caffe_illu/tools/caffe

$CBIN train --solver=./solver.prototxt --log_dir log/ $@ -gpu 0,2 -weights snapshots/SRResNet-MSE_iter_70000.caffemodel
