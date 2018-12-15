#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/bin/bash

# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

. ${MLSUITE_ROOT}/overlaybins/setup.sh 

for DSP_WIDTH in 28 56 96; do
    if [ ${DSP_WIDTH} == 28 ]; then
        MEM=4
    elif [ ${DSP_WIDTH} == 56 ]; then
        MEM=6
    elif [ ${DSP_WIDTH} == 96 ]; then
        MEM=9
    fi
    DDR=256

    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/squeezenet/fp32/squeezenet_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/squeezenet/fp32/squeezenet_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/squeezenet/fp32/squeezenet.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m ${MEM} \
        -d ${DDR}
done

