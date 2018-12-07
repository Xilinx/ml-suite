#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/bin/bash

# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
    MLSUITE_ROOT=../../..
fi

. ${MLSUITE_ROOT}/overlaybins/setup.sh

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/mobilenet/fp32/mobilenet_without_bn_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1024 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/mobilenet/fp32/mobilenet_without_bn.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --img_input_scale 0.017 \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/examples/classification/dog.jpg \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/mobilenet/mobilenet_without_bn_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/mobilenet/fp32/mobilenet_without_bn_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1024 \
#            --datadir $MLSUITE_ROOT/models/caffe/mobilenet/fp32/mobilenet_without_bn.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --img_input_scale 0.017 \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/examples/classification/dog.jpg \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/mobilenet/mobilenet_without_bn_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/mobilenet/fp32/mobilenet_without_bn_no_dw_inner_product_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1024 \
#            --datadir $MLSUITE_ROOT/models/caffe/mobilenet/fp32/mobilenet_without_bn_no_dw_inner_product.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --img_input_scale 0.017 \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/examples/classification/dog.jpg \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/mobilenet/mobilenet_without_bn_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

for BITWIDTH in 8; do
    for DSP_WIDTH in 56; do
        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/mobilenet/fp32/mobilenet_without_bn_deploy_${DSP_WIDTH}.cmds \
            --fpgaoutsz 1024 \
            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/mobilenet/fp32/mobilenet_without_bn.caffemodel_data \
            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
            --img_input_scale 0.017 \
            --xlnxlib $LIBXDNN_PATH \
            --images $MLSUITE_ROOT/examples/classification/dog.jpg \
            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/mobilenet/mobilenet_without_bn_quantized_int${BITWIDTH}_deploy.json \
            --firstfpgalayer conv1
    done
done

