#!/bin/bash

for DSP_WIDTH in 28 56; do
    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/vgg16/fp32/vgg16_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/vgg16/fp32/vgg16_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/vgg16/fp32/vgg16.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m 4 \
        -d 16
done

