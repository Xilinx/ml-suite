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

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_baseline_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1024 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_baseline.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_baseline_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1/7x7_s2
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v1_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 452 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v1.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_pruned_v1_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1/7x7_s2
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v2_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 222 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v2.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_pruned_v2_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1/7x7_s2
#    done
#done

for BITWIDTH in 8; do
    for DSP_WIDTH in 56; do
        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v3_deploy_${DSP_WIDTH}.cmds \
            --fpgaoutsz 336 \
            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_v1_pruned_v3.caffemodel_data \
            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
            --xlnxlib $LIBXDNN_PATH \
            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_v1_pruned_v3_quantized_int${BITWIDTH}_deploy.json \
            --firstfpgalayer conv1/7x7_s2
    done
done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_baseline_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1024 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_baseline.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_baseline_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1/7x7_s2
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_pruned_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 284 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/inception_pruned.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/inception_pruned_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1/7x7_s2
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_baseline_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 2048 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_baseline.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_baseline_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_pruned_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1844 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_pruned.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_pruned_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_baseline_with_scale_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 2048 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_baseline_with_scale.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_baseline_with_scale_quantized_int${BITWIDTH}_deploy.json \
#            --firstfpgalayer conv1
#    done
#done

#for BITWIDTH in 8; do
#    for DSP_WIDTH in 56; do
#        python $MLSUITE_ROOT/examples/classification/mp_classify.py \
#            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
#            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_pruned_with_scale_deploy_${DSP_WIDTH}.cmds \
#            --fpgaoutsz 1844 \
#            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/deephi/fp32/resnet50_pruned_with_scale.caffemodel_data \
#            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
#            --xlnxlib $LIBXDNN_PATH \
#            --images $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
#            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/deephi/resnet50_pruned_with_scale_quantized_int${BITWIDTH}_deploy.json \
#            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
#            --firstfpgalayer conv1
#    done
#done

