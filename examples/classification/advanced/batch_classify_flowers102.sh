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

export XBLAS_EMIT_PROFILING_INFO=1

for BITWIDTH in 8; do
    for DSP_WIDTH in 56; do
        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/flowers102/fp32/bvlc_googlenet_without_lrn_deploy_${DSP_WIDTH}.cmds \
            --outsz 102 \
            --fpgaoutsz 1024 \
            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/flowers102/fp32/bvlc_googlenet_without_lrn.caffemodel_data \
            --labels $MLSUITE_ROOT/models/data/flowers102/synset_words.txt \
            --xlnxlib $LIBXDNN_PATH \
            --images $MLSUITE_ROOT/models/data/flowers102/flowers102_img_val \
            --golden $MLSUITE_ROOT/models/data/flowers102/val.txt \
            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/flowers102/bvlc_googlenet_without_lrn_quantized_int${BITWIDTH}_deploy.json \
            --firstfpgalayer conv1/7x7_s2
    done
done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/flowers102/fp32/resnet50_without_bn_deploy_${DSP_WIDTH}.cmds \
#            --outsz 102 \
#            --fpgaoutsz 1024 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/flowers102/fp32/resnet50_without_bn.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/flowers102/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/flowers102/flowers102_img_val \
#            --golden $MLSUITE_ROOT/models/data/flowers102/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/flowers102/resnet50_without_bn_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

