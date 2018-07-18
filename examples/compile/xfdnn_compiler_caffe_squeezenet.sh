#!/bin/bash

for DSP_WIDTH in 28 56; do
    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
        -n $MLSUITE_ROOT/models/caffe/squeezenet/fp32/squeezenet_deploy.prototxt \
        -g $MLSUITE_ROOT/examples/compile/work/caffe/squeezenet/fp32/squeezenet_deploy_${DSP_WIDTH}.cmds \
        -w $MLSUITE_ROOT/models/caffe/squeezenet/fp32/squeezenet.caffemodel \
        -s all \
        -i ${DSP_WIDTH} \
        -m 4 \
        -d 0
done

