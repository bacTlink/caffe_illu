#!/usr/bin/env sh
set -e

CBIN=/home/linzehui/illu/build/caffe_illu/tools/caffe

$CBIN train --solver=./solver_srresnet_auto.prototxt --log_dir log/ $@ -gpu 0,1 --snapshot snapshots/BOX-colored_iter_189201.solverstate
