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

mkdir -p snapshots/v5/
mkdir -p log

cd ./scripts/
python ./build_resnet.py --output $THIS_DIR \
                         --stage_link \
                         --batch 20 \
                         --prefix /data3/lzh/1000x224x224_largeview/ 

cd $THIS_DIR
./train.sh $@
