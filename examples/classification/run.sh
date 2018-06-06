#!/usr/bin/env bash

DEVICE=$1
TEST_TO_RUN=$2
PE_CFG=$3
BITWIDTH=$4

if [ "$PE_CFG" == "med" ]; then
  DSP_WIDTH=28
else
  DSP_WIDTH=56
fi

. ../../overlaybins/$DEVICE/setup.sh

echo "=============== pyXDNN ============================="

if [ "$TEST_TO_RUN" == "test_classify" ]; then
  #################
  # single image 
  #################

  python test_classify.py --xclbin $XCLBIN_PATH/xdnn_${DSP_WIDTH}_${BITWIDTH}b.xclbin --netcfg ./data/googlenet_v1_${DSP_WIDTH}.cmd --fpgaoutsz 1024 --datadir ./data/googlenet_v1_data --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --quantizecfg ./data/googlenet_v1_${BITWIDTH}b.json --firstfpgalayer conv1/7x7_s2 --images dog.jpg --useblas

elif [ "$TEST_TO_RUN" == "batch_classify" ]; then
  ############################
  # multi-process streaming 
  ############################

  python batch_classify.py --xclbin $XCLBIN_PATH/xdnn_${DSP_WIDTH}_${BITWIDTH}b.xclbin --netcfg ./data/googlenet_v1_${DSP_WIDTH}.cmd --fpgaoutsz 1024 --datadir ./data/googlenet_v1_data --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --imagedir imagenet_val/ --useblas --golden gold.txt --quantizecfg ./data/googlenet_v1_${BITWIDTH}b.json --firstfpgalayer conv1/7x7_s2

elif [ "$TEST_TO_RUN" == "multinet" ]; then

  ###########################
  # multi-PE multi-network 
  ###########################

  python test_classify_async_multinet.py --xclbin $XCLBIN_PATH/xdnn_28_16b.xclbin --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --jsoncfg data/multinet.json

else
  echo "Hello, goodbye!"
fi
