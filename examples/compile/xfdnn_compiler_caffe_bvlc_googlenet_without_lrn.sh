#!/bin/bash

for DSP_WIDTH in 28 56; do
    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m 4 \
        -d 0
done

