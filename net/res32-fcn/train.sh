#!/usr/bin/env sh
set -e

CBIN=../../build/tools/caffe

$CBIN train --solver=./solver.prototxt --log_dir log/ $@
