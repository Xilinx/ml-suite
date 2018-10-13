#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "./run.sh --platform <platform> --test <test> --m <model> --k <kcfg> --b <bitwidth>"
  echo "./run.sh  -p        <platform>  -t    <test>  -m <model>  -k <kcfg>  -b <bitwidth>"
  echo "<platform> : 1525 / xbb-u200 / xbb-u250 / aws / nimbix"
  echo "<test>     : test_classify / batch_classify"
  echo "<kcfg>     : med / large"
  echo "<bitwidth> : 8 / 16"
  echo "Some tests require a directory of images to process."
  echo "To process a directory in a non-standard location use -d <directory> or --directory <directory>"
  echo "Some tests require a batchSize argument to know how many images to load simultaneously."
  echo "To provide batchSize use --batchsize <batchsize>"
}

# Default
# PLATFORM is REQUIRED
TEST="test_classify"
MODEL="googlenet_v1"
KCFG="large"
BITWIDTH="8"
ACCELERATOR="0"
# This variable is used for batch classify
# You should put the imagenet validation set in the below folder
DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
BATCHSIZE=4
VERBOSE=0

# Parse Options
OPTS=`getopt -o p:t:m:k:b:d:s:a:vh --long platform:,test:,model:,kcfg:,bitwidth:,directory:,batchsize:,accelerator:,verbose,help -n "$0" -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi
while true
do
  case "$1" in
    -p|--platform    ) PLATFORM="$2"   ; shift 2 ;;
    -t|--test        ) TEST="$2"       ; shift 2 ;;
    -m|--model       ) MODEL="$2"      ; shift 2 ;;
    -k|--kcfg        ) KCFG="$2"       ; shift 2 ;;
    -b|--bitwidth    ) BITWIDTH="$2"   ; shift 2 ;;
    -d|--directory   ) DIRECTORY="$2"  ; shift 2 ;;
    -s|--batchsize   ) BATCHSIZE="$2"  ; shift 2 ;;
    -a|--accelerator ) ACCELERATOR="$2"; shift 2 ;;
    -v|--verbose     ) VERBOSE=1       ; shift 1 ;;
    -h|--help        ) usage           ; exit  1 ;;
     *) break ;;
  esac
done

if [ -z $PLATFORM ]; then
  echo "Error: Please specify a platform with -p or --platform";usage;exit 1
fi

# Verbose Debug Profiling Prints
export XBLAS_EMIT_PROFILING_INFO=$VERBOSE

# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi
. ${MLSUITE_ROOT}/overlaybins/setup.sh $PLATFORM

# Determine XCLBIN and DSP_WIDTH
XCLBIN="not_found.xclbin"
if [ "$KCFG" == "med" ]; then
  DSP_WIDTH=28
  XCLBIN=overlay_1.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_0.xclbin
  fi
else # if [ "$KCFG" == "large" ]; then
  DSP_WIDTH=56
  XCLBIN=overlay_3.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_2.xclbin
  fi
fi

# Determine FPGAOUTSZ and FLAYER which depend upon model
if [ "$MODEL" == "googlenet_v1" ]; then
  FPGAOUTSZ=1024
  FLAYER="conv1/7x7_s2"
elif [ "$MODEL" == "mobilenet" ]; then
  FPGAOUTSZ=1024
  FLAYER="conv1"
elif [ "$MODEL" == "resnet50" ]; then
  FPGAOUTSZ=2048
  FLAYER="conv1"
fi

echo -e "\nRunning:\n Test: $TEST\n Model: $MODEL\n Fpgaoutsz: $FPGAOUTSZ\n Platform: $PLATFORM\n Xclbin: $XCLBIN\n Kernel Config: $KCFG\n Precision: $BITWIDTH\n Accelerator: $ACCELERATOR\n"

# Call appropriate python script
if [ "$TEST" == "test_classify" ]; then
  ####################
  # single image test
  ####################
  python test_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/${MODEL}_${DSP_WIDTH}.json --fpgaoutsz $FPGAOUTSZ --datadir ./data/${MODEL}_data --labels ./synset_words.txt --xlnxlib $LIBXDNN_PATH --quantizecfg ./data/${MODEL}_${BITWIDTH}b.json --firstfpgalayer $FLAYER --images dog.jpg --PE $ACCELERATOR 

elif [ "$TEST" == "batch_classify" ]; then
  ############################
  # multi-process streaming 
  ############################
  python batch_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/${MODEL}_${DSP_WIDTH}.json --fpgaoutsz $FPGAOUTSZ --datadir ./data/${MODEL}_data --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --imagedir $DIRECTORY --golden gold.txt --quantizecfg ./data/${MODEL}_${BITWIDTH}b.json --firstfpgalayer $FLAYER --batchSize $BATCHSIZE

elif [ "$TEST" == "perpetual" ]; then
  ############################
  # multi-process streaming 
  ############################
  export XBLAS_EMIT_PROFILING_INFO=1
  python batch_classify.py --xclbin $XCLBIN_PATH/overlay_0.xclbin --netcfg ./data/${MODEL}_28.json --fpgaoutsz $FPGAOUTSZ --datadir ./data/${MODEL}_data --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --imagedir $DIRECTORY --golden gold.txt --quantizecfg ./data/${MODEL}_8b.json --firstfpgalayer $FLAYER --batchSize 8 --zmqpub True --perpetual True

#elif [ "$TEST" == "streaming_classify" ]; then
#  ############################
#  # multi-process streaming 
#  ############################
#  python streaming_classify.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ./data/${MODEL}_${DSP_WIDTH}.json --fpgaoutsz $FPGAOUTSZ --datadir ./data/${MODEL}_data --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --imagedir $DIRECTORY --golden gold.txt --quantizecfg ./data/${MODEL}_${BITWIDTH}b.json --firstfpgalayer $FLAYER --batchSize $BATCHSIZE #--bypassLoad #--numImages 4096 --perpetual #--zmqpub True --perpetual True
#
elif [ "$TEST" == "multinet" ]; then
  ###########################
  # multi-PE multi-network (Run two different networks simultaneously) 
  ###########################
  python test_classify_async_multinet.py --xclbin $XCLBIN_PATH/overlay_1.xclbin --labels synset_words.txt --xlnxlib $LIBXDNN_PATH --jsoncfg data/multinet.json

else
  echo "Test was improperly specified..."
fi
