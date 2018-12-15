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
#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_baseline_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_baseline_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_baseline.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v1_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_pruned_v1_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v1.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v2_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_pruned_v2_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v2.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v3_deploy.prototxt \
        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_pruned_v3_quantized_int${BITWIDTH}_deploy.json \
        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v3.caffemodel \
        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
        --calibration_size 32 \
        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
        --dims 3,224,224 \
        --transpose 2,0,1 \
        --channel_swap 2,1,0 \
        --raw_scale 255.0 \
        --mean_value 104.0,117.0,123.0 \
        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_baseline_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_baseline_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_baseline.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_pruned_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_pruned_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_pruned.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_baseline_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_pruned_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline_with_scale_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_baseline_with_scale_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline_with_scale.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0

#    python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.py \
#        --deploy_model $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned_with_scale_deploy.prototxt \
#        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_pruned_with_scale_quantized_int${BITWIDTH}_deploy.json \
#        --weights $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned_with_scale.caffemodel \
#        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
#        --calibration_size 32 \
#        --bitwidths ${BITWIDTH},${BITWIDTH},${BITWIDTH} \
#        --dims 3,224,224 \
#        --transpose 2,0,1 \
#        --channel_swap 2,1,0 \
#        --raw_scale 255.0 \
#        --mean_value 104.0,117.0,123.0 \
#        --input_scale 1.0
done

