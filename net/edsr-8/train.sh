#!/usr/bin/env sh
set -e

CBIN=/home/linzehui/illu/build/caffe_illu/tools/caffe

$CBIN train --solver=./solver_srresnet_auto.prototxt --log_dir log/ $@ -gpu 0,1 --weights snapshots/BOX_iter_209241.caffemodel
