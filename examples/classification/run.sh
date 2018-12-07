#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "./run.sh --platform <platform> --test <test> --model <model> --kcfg <kcfg> --bitwidth <bitwidth>"
  echo "./run.sh  -p        <platform>  -t    <test>  -m <model>  -k <kcfg>  -b <bitwidth>"
  echo "<platform> : 1525 / 1525-ml / alveo-u200 / alveo-u200-ml / alveo-u250 / aws / nimbix"
  echo "<test>     : test_classify / streaming_classify"
  echo "<kcfg>     : med / large / v3"
  echo "<bitwidth> : 8 / 16"
  echo "<compilerOpt> : autoAllOpt / latency / throughput"
  echo "Some tests require a directory of images to process."
  echo "To process a directory in a non-standard location use -d <directory> or --directory <directory>"
  echo "Some tests require a batchSize argument to know how many images to load simultaneously."
  echo "To provide batchSize use --batchsize <batchsize>"
  echo "-c allows to choose compiler optimization, for example, latency or throughput or autoAllOptimizations."
  echo "-g runs entire test providing top-1 and top-5 results"

}

# Default
TEST="test_classify"
MODEL="googlenet_v1"
KCFG="large"
BITWIDTH="8"
ACCELERATOR="0"
BATCHSIZE=-1
VERBOSE=0
PERPETUAL=0
IMG_INPUT_SCALE=1.0
# These variables are used in case there multiple FPGAs running in parallel
NUMDEVICES=1
DEVICEID=0
NUMPREPPROC=4
COMPILEROPT="autoAllOpt.json"
MODE="throughput"
# Parse Options
OPTS=`getopt -o p:t:m:k:b:d:s:a:n:i:c:y:gvxh --long platform:,test:,model:,kcfg:,bitwidth:,directory:,numdevices:,deviceid:,batchsize:,compilerOpt:,numprepproc,checkaccuracy,verbose,perpetual,help -n "$0" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi
  
while true
do
  case "$1" in
    -p|--platform    ) MLSUITE_PLATFORM="$2" ; shift 2 ;;
    -t|--test        ) TEST="$2"             ; shift 2 ;;
    -m|--model       ) MODEL="$2"            ; shift 2 ;;
    -k|--kcfg        ) KCFG="$2"             ; shift 2 ;;
    -b|--bitwidth    ) BITWIDTH="$2"         ; shift 2 ;;
    -d|--directory   ) DIRECTORY="$2"        ; shift 2 ;;
    -s|--batchsize   ) BATCHSIZE="$2"        ; shift 2 ;;
    -a|--accelerator ) ACCELERATOR="$2"      ; shift 2 ;;
    -n|--numdevices  ) NUMDEVICES="$2"       ; shift 2 ;;
    -i|--deviceid    ) DEVICEID="$2"         ; shift 2 ;;
    -c|--compilerOpt ) COMPILEROPT="$2"      ; shift 2 ;;
    -y|--numprepproc ) NUMPREPPROC="$2"      ; shift 2 ;;
    -g|--checkaccuracy ) GOLDEN=gold.txt     ; shift 1 ;;
    -v|--verbose     ) VERBOSE=1             ; shift 1 ;;
    -x|--perpetual   ) PERPETUAL=1             ; shift 1 ;;
    -h|--help        ) usage                 ; exit  1 ;;
     *) break ;;
  esac
done

# Verbose Debug Profiling Prints
# Note, the VERBOSE mechanic here is not working
# Its always safer to set this manually
export XBLAS_EMIT_PROFILING_INFO=1
# To be fixed
export XBLAS_EMIT_PROFILING_INFO=$VERBOSE
#export XDNN_VERBOSE=1
# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

. ${MLSUITE_ROOT}/overlaybins/setup.sh ${MLSUITE_PLATFORM}

# Determine FPGAOUTSZ which depend upon model
if [ "$MODEL" == "googlenet_v1" ]; then
  FPGAOUTSZ=1024
elif [ "$MODEL" == "mobilenet" ]; then
  FPGAOUTSZ=1024
  IMG_INPUT_SCALE=0.017
elif [ "$MODEL" == "resnet50" ]; then
  FPGAOUTSZ=2048
elif [ "$MODEL" == "resnet101" ]; then
  FPGAOUTSZ=2048
fi

# Determine XCLBIN and DSP_WIDTH
XCLBIN="not_found.xclbin"
WEIGHTS=./data/${MODEL}_data
if [ "$KCFG" == "med" ]; then
  DSP_WIDTH=28
  XCLBIN=overlay_1.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_0.xclbin
  fi
  NETCFG=./data/${MODEL}_${DSP_WIDTH}.json
  QUANTCFG=./data/${MODEL}_${BITWIDTH}b.json
elif [ "$KCFG" == "large" ]; then
  DSP_WIDTH=56
  XCLBIN=overlay_3.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_2.xclbin
  fi
  NETCFG=./data/${MODEL}_${DSP_WIDTH}.json
  QUANTCFG=./data/${MODEL}_${BITWIDTH}b.json
elif [ "$KCFG" == "v3" ]; then
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_4.xclbin
  elif [ "$BITWIDTH" == "16" ]; then
    XCLBIN=overlay_5.xclbin
  fi

  if [ $COMPILEROPT == "latency" ] && [ $MODEL == "googlenet_v1" ]; then
    COMPILEROPT=latency.cmds
    export XDNN_LATENCY_OPTIMIZED=1
    WEIGHTS=./data/${MODEL}_data
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_xdnnv3.json 

  elif [ $COMPILEROPT == "throughput" ] && [ $MODEL == "googlenet_v1" ]; then
    COMPILEROPT=throughput.json
    export XDNN_THROUGHPUT_OPTIMIZED=1
    WEIGHTS=./data/${MODEL}_data
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_xdnnv3.json 

  elif [ $COMPILEROPT == "throughput" ] && [ $MODEL == "resnet50" ]; then
    COMPILEROPT=throughput.json
    export XDNN_RESNET_THROUGHPUT_OPTIMIZED=1
    WEIGHTS=./data/${MODEL}_tensorflow_data
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_tensorflow_xdnnv3.json 
  
  else
    if [ $COMPILEROPT == "latency" ]; then
      COMPILEROPT=latency.json
    else
      COMPILEROPT="autoAllOpt.json"
    fi
    WEIGHTS=./data/${MODEL}_tensorflow_data
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_tensorflow_xdnnv3.json
  fi

else
  echo "Unsupported kernel config $KCFG"
  exit 1
fi

echo -e "Running:\n Test: $TEST\n Model: $MODEL\n Fpgaoutsz: $FPGAOUTSZ\n Platform: $MLSUITE_PLATFORM\n Xclbin: $XCLBIN\n Kernel Config: $KCFG\n Precision: $BITWIDTH\n Accelerator: $ACCELERATOR\n"

BASEOPT="--xclbin $XCLBIN_PATH/$XCLBIN 
         --netcfg $NETCFG 
         --fpgaoutsz $FPGAOUTSZ 
         --datadir $WEIGHTS 
         --labels ./synset_words.txt 
         --quantizecfg $QUANTCFG 
         --img_input_scale $IMG_INPUT_SCALE 
         --batch_sz $BATCHSIZE"

if [ ! -z $GOLDEN ]; then
  BASEOPT+=" --golden $GOLDEN"
fi

# Build options for appropriate python script
####################
# single image test
####################
if [ "$TEST" == "test_classify" ]; then
  TEST=test_classify.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
    
  fi
  BASEOPT+=" --images $DIRECTORY"
############################
# multi-process streaming 
############################  
elif [ "$TEST" == "streaming_classify" ]; then
  TEST=mp_classify.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --numprepproc $NUMPREPPROC"
  if [ "$PERPETUAL" == 1 ]; then 
    BASEOPT+=" --zmqpub --perpetual --deviceID $DEVICEID"
  fi 
###########################
# multi-PE multi-network (Run two different networks simultaneously)
# runs with 8 bit quantization for now
###########################
elif [ "$TEST" == "multinet" ]; then
  TEST=test_classify_async_multinet.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
  fi
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --dsp $DSP_WIDTH --jsoncfg data/multinet.json"
else
  echo "Test was improperly specified..."
  exit 1
fi

python $TEST $BASEOPT
