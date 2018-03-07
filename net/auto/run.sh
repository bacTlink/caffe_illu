#!/bin/bash

################################################################
# File Name: run.sh
# Author: gaoyu
# Mail: 1400012705@pku.edu.cn
# Created Time: 2018-03-07 11:30:15
################################################################

set -e

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
cd $THIS_DIR

mkdir -p snapshots/v4/
mkdir -p log

cd ./scripts/
python ./build_resnet.py --output $THIS_DIR 

cd $THIS_DIR
./train.sh $@
