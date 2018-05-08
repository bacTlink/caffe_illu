#!/usr/bin/env sh
set -e

CBIN=/home/linzehui/illu/build/caffe_illu/tools/caffe

$CBIN train --solver=./solver_srresnet_auto.prototxt --log_dir log/ $@ -gpu 1 --snapshot snapshots/DIAMOND-colored_iter_271305.solverstate
