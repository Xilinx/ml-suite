#!/usr/bin/env bash

DEVICE=$1
TEST_TO_RUN=$2
PE_CFG=$3
BITWIDTH=$4

XCLBIN="not_found.xclbin"
if [ "$PE_CFG" == "med" ]; then
  DSP_WIDTH=28
  XCLBIN=overlay_1.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_0.xclbin
  fi
else # if [ "$PE_CFG" == "large" ]; then
  DSP_WIDTH=56
  XCLBIN=overlay_3.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_2.xclbin
  fi
fi

if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

. ${MLSUITE_ROOT}/overlaybins/setup.sh $DEVICE

# This variable is used for batch classify
# You should put the imagenet validation set in the below folder
IMAGEDIR=../../models/data/ilsvrc12/ilsvrc12_img_val

echo "=============== pyXDNN ============================="

if [ "$TEST_TO_RUN" == "test_classify" ]; then
  #################
  # single image 
  #################

  python test_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/googlenet_v1_${DSP_WIDTH}.cmd --fpgaoutsz 1024 --datadir ./data/googlenet_v1_data --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --quantizecfg ./data/googlenet_v1_${BITWIDTH}b.json --firstfpgalayer conv1/7x7_s2 --images dog.jpg --useblas

elif [ "$TEST_TO_RUN" == "test_classify_resnet" ]; then
  #################
  # single image
  # resnet using global quantization
  #################

  #python test_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/resnet50_${DSP_WIDTH}.cmd --fpgaoutsz 2048 --datadir ./data/resnet50_data --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --quantizecfg ./data/resnet50_${BITWIDTH}b.json --firstfpgalayer conv1 --images dog.jpg --useblas
  python test_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/resnet50_${DSP_WIDTH}.cmd --fpgaoutsz 2048 --datadir ./data/resnet50_data --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --images dog.jpg --useblas

elif [ "$TEST_TO_RUN" == "test_classify_mobilenet" ]; then
  #################
  # single image
  # mobilenet 16bit
  #################
  export XBLAS_EMIT_PROFILING_INFO=0
  python test_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/mobilenet_${DSP_WIDTH}.cmd --fpgaoutsz 1024 --datadir ./data/mobilenet_data --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --quantizecfg ./data/mobilenet_16b.json --firstfpgalayer conv1 --images dog.jpg --useblas --transform mobilenet

elif [ "$TEST_TO_RUN" == "batch_classify" ]; then
  ############################
  # multi-process streaming 
  ############################

  export XBLAS_EMIT_PROFILING_INFO=1
  python batch_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/googlenet_v1_${DSP_WIDTH}.cmd --fpgaoutsz 1024 --datadir ./data/googlenet_v1_data --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --imagedir $IMAGEDIR --useblas --golden gold.txt --quantizecfg ./data/googlenet_v1_${BITWIDTH}b.json --firstfpgalayer conv1/7x7_s2 #--zmqpub True --perpetual True

elif [ "$TEST_TO_RUN" == "batch_classify_mobilenet" ]; then
  ############################
  # multi-process streaming 
  ############################

  export XBLAS_EMIT_PROFILING_INFO=0
  python batch_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/mobilenet_${DSP_WIDTH}.cmd --fpgaoutsz 1024 --datadir ./data/mobilenet_data --imagedir $IMAGEDIR --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --quantizecfg ./data/mobilenet_16b.json --firstfpgalayer conv1 --golden gold.txt --useblas --transform mobilenet

elif [ "$TEST_TO_RUN" == "multinet" ]; then

  ###########################
  # multi-PE multi-network 
  ###########################

  python test_classify_async_multinet.py --xclbin $XCLBIN_PATH/xdnn_v2_32x28_4pe_16b_4mb_bank21.xclbin --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --jsoncfg data/multinet.json

else
  echo "Hello, goodbye!"
fi
