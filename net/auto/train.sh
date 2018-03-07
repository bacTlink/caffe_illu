#!/usr/bin/env sh
set -e

CBIN=/home/gaoyu/caffe_illu/build/tools/caffe

$CBIN train --solver=./solver.prototxt --log_dir log/ $@
