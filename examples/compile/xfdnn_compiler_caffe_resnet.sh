#!/bin/bash

for DSP_WIDTH in 28 56; do
    if [ ${DSP_WIDTH} == 28 ]; then
        DDR=0
    elif [ ${DSP_WIDTH} == 56 ]; then
        DDR=16
    fi

    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/resnet/fp32/resnet50_without_bn_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/resnet/fp32/resnet50_without_bn_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/resnet/fp32/resnet50_without_bn.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m 4 \
        -d ${DDR}
    
    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/resnet/fp32/resnet101_without_bn_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/resnet/fp32/resnet101_without_bn_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/resnet/fp32/resnet101_without_bn.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m 4 \
        -d ${DDR}
    
    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/resnet/fp32/resnet152_without_bn_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/resnet/fp32/resnet152_without_bn_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/resnet/fp32/resnet152_without_bn.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m 4 \
        -d ${DDR}
done

