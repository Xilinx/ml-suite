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

for BITWIDTH in 16 8; do
    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
        --deploy_model $MLSUITE_ROOT/models/caffe/mobilenet/fp32/mobilenet_without_bn_deploy.prototxt \
        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/mobilenet/mobilenet_without_bn_quantized_int${BITWIDTH}_deploy.json \
        --weights $MLSUITE_ROOT/models/caffe/mobilenet/fp32/mobilenet_without_bn.caffemodel \
        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
        --calibration_size 32 \
        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
        --dims 3,224,224 \
        --transpose 2,0,1 \
        --channel_swap 2,1,0 \
        --raw_scale 255.0 \
        --mean_value 103.94,116.78,123.68 \
        --input_scale 0.017
done

