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

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_baseline_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_baseline_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_baseline.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v1_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v1_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v1.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v2_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v2_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v2.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v3_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v3_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_v1_pruned_v3.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m ${MEM} \
        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_baseline_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_baseline_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_baseline.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m 4 \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_pruned_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_pruned_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/inception_pruned.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_baseline_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_pruned_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline_with_scale_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_baseline_with_scale_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_baseline_with_scale.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}

#    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
#        -n $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned_with_scale_deploy.prototxt \
#        -g $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_pruned_with_scale_deploy_${DSP_WIDTH}.cmds \
#        -w $MLSUITE_ROOT/models/caffe/deephi/fp32/resnet50_pruned_with_scale.caffemodel \
#        -s all \
#        -i ${DSP_WIDTH} \
#        -m ${MEM} \
#        -d ${DDR}
done

