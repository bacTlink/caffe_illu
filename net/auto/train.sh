#!/usr/bin/env sh
set -e

CBIN=/home/gaoyu/caffe_illu/build/tools/caffe

$CBIN train --solver=./solver_no_link_relu.prototxt --log_dir log/pooling_box/ $@
