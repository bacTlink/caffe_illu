#!/usr/bin/env sh
set -e

CBIN=../build/tools/caffe

$CBIN train --solver=./solver.prototxt --gpu 1 $@
