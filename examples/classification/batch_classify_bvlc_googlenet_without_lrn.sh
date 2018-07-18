#!/bin/bash

for BITWIDTH in 16 8; do
    for DSP_WIDTH in 28; do
        python batch_classify.py \
            --xclbin $XCLBIN_PATH/xdnn_v2_32x${DSP_WIDTH}_$((112 / ${DSP_WIDTH}))pe_${BITWIDTH}b_$((2 + ${DSP_WIDTH} / 14))mb_bank21.xclbin \
            --netcfg $MLSUITE_ROOT/examples/compile/work/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy_${DSP_WIDTH}.cmds \
            --fpgaoutsz 1024 \
            --datadir $MLSUITE_ROOT/examples/compile/work/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel_data \
            --labels $MLSUITE_ROOT/models/data/ilsvrc12/synset_words.txt \
            --xlnxlib $LIBXDNN_PATH \
            --imagedir $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_val \
            --useblas \
            --golden $MLSUITE_ROOT/models/data/ilsvrc12/val.txt \
            --quantizecfg $MLSUITE_ROOT/examples/quantize/work/caffe/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_int${BITWIDTH}_deploy.json \
            --firstfpgalayer conv1/7x7_s2
    done
done

