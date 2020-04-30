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
  echo "<test>     : gRPC"
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
TEST="gRPC"
MODEL="googlenet_v1"
KCFG="v3"
BITWIDTH="8"
ACCELERATOR="0"
BATCHSIZE=-1
VERBOSE=0
FRODO=0
ZELDA=0
PERPETUAL=0
IMG_INPUT_SCALE=1.0
# These variables are used in case there multiple FPGAs running in parallel
NUMDEVICES=1
NUMSTREAMS=8
DEVICEID=0
NUMPREPPROC=4
COMPILEROPT="autoAllOpt.json"
# Parse Options
OPTS=`getopt -o p:t:m:k:b:d:s:a:n:ns:i:c:y:gvzfxh --long platform:,test:,model:,kcfg:,bitwidth:,directory:,numdevices:,numstreams:,deviceid:,batchsize:,compilerOpt:,numprepproc,checkaccuracy,verbose,zelda,frodo,perpetual,help -n "$0" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi

while true
do
  case "$1" in
    -p |--platform      ) MLSUITE_PLATFORM="$2" ; shift 2 ;;
    -t |--test          ) TEST="$2"             ; shift 2 ;;
    -m |--model         ) MODEL="$2"            ; shift 2 ;;
    -k |--kcfg          ) KCFG="$2"             ; shift 2 ;;
    -b |--bitwidth      ) BITWIDTH="$2"         ; shift 2 ;;
    -d |--directory     ) DIRECTORY="$2"        ; shift 2 ;;
    -s |--batchsize     ) BATCHSIZE="$2"        ; shift 2 ;;
    -a |--accelerator   ) ACCELERATOR="$2"      ; shift 2 ;;
    -n |--numdevices    ) NUMDEVICES="$2"       ; shift 2 ;;
    -ns|--numstreams    ) NUMSTREAMS="$2"       ; shift 2 ;;
    -i |--deviceid      ) DEVICEID="$2"         ; shift 2 ;;
    -c |--compilerOpt   ) COMPILEROPT="$2"      ; shift 2 ;;
    -y |--numprepproc   ) NUMPREPPROC="$2"      ; shift 2 ;;
    -g |--checkaccuracy ) GOLDEN=gold.txt       ; shift 1 ;;
    -v |--verbose       ) VERBOSE=1             ; shift 1 ;;
    -f |--frodo         ) FRODO=1               ; shift 1 ;;
    -z |--zelda         ) ZELDA=1               ; shift 1 ;;
    -x |--perpetual     ) PERPETUAL=1           ; shift 1 ;;
    -cn|--customnet     ) CUSTOM_NETCFG="$2"    ; shift 2 ;;
    -cq|--customquant   ) CUSTOM_QUANTCFG="$2"  ; shift 2 ;;
    -cw|--customwts     ) CUSTOM_WEIGHTS="$2"   ; shift 2 ;;
    -h |--help          ) usage                 ; exit  1 ;;
     *) break ;;
  esac
done

# Verbose Debug Profiling Prints
# Note, the VERBOSE mechanic here is not working
# Its always safer to set this manually
export XBLAS_EMIT_PROFILING_INFO=1
# To be fixed
export XBLAS_EMIT_PROFILING_INFO=$VERBOSE
export XDNN_VERBOSE=$VERBOSE
# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

. ${MLSUITE_ROOT}/overlaybins/setup.sh ${MLSUITE_PLATFORM}

if [ "$MODEL" == "mobilenet" ]; then
  IMG_INPUT_SCALE=0.017
fi

# Determine XCLBIN and DSP_WIDTH
XCLBIN="not_found.xclbin"
WEIGHTS=./data/${MODEL}_data.h5
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
  DSP_WIDTH=96
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_4.xclbin
  elif [ "$BITWIDTH" == "16" ]; then
    XCLBIN=overlay_5.xclbin
  fi

  if [ $COMPILEROPT == "latency" ] && [ $MODEL == "googlenet_v1" ]; then
    COMPILEROPT=latency.json
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_xdnnv3.json

  elif [ $COMPILEROPT == "throughput" ] && [ $MODEL == "googlenet_v1" ]; then
    COMPILEROPT=throughput.json
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_xdnnv3.json

  elif [ $COMPILEROPT == "throughput" ] && [ $MODEL == "resnet50" ]; then
    COMPILEROPT=throughput.json
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_xdnnv3.json

  else
    if [ $COMPILEROPT == "latency" ]; then
      COMPILEROPT=latency.json
    else
      COMPILEROPT="autoAllOpt.json"
    fi
    NETCFG=./data/${MODEL}_${BITWIDTH}b_${COMPILEROPT}
    QUANTCFG=./data/${MODEL}_${BITWIDTH}b_xdnnv3.json
  fi

else
  echo "Unsupported kernel config $KCFG"
  exit 1
fi

if [ ! -z $CUSTOM_NETCFG ]; then
  NETCFG=$CUSTOM_NETCFG
fi
if [ ! -z $CUSTOM_WEIGHTS ]; then
  WEIGHTS=$CUSTOM_WEIGHTS
fi
if [ ! -z $CUSTOM_QUANTCFG ]; then
  QUANTCFG=$CUSTOM_QUANTCFG
fi

echo -e "Running:\n Test: $TEST\n Model: $MODEL\n Platform: $MLSUITE_PLATFORM\n Xclbin: $XCLBIN\n Kernel Config: $KCFG\n Precision: $BITWIDTH\n Accelerator: $ACCELERATOR\n"

BASEOPT="--xclbin $XCLBIN_PATH/$XCLBIN
         --netcfg $NETCFG
         --weights $WEIGHTS
         --labels ./synset_words.txt
         --quantizecfg $QUANTCFG
         --img_input_scale $IMG_INPUT_SCALE
         --batch_sz $BATCHSIZE"

if [ ! -z $GOLDEN ]; then
  BASEOPT+=" --golden $GOLDEN"
fi

# Build options for appropriate python script
############################
# gPRC server
############################
if [[ "$TEST" == "gRPC"* ]]; then
  BASEOPT+=" --images ."

  TEST=server.py
fi

if [ "$TEST" == "classify_cpp" ]; then
  ./classify.exe $BASEOPT_CPP
elif [ "$ZELDA" -eq "1" ]; then
  echo python $TEST $BASEOPT
  gdb --args python $TEST $BASEOPT
elif [ "$FRODO" -eq "1" ]; then
  echo python $TEST $BASEOPT
  valgrind --tool=memcheck --leak-check=full --show-reachable=yes --log-file="valgrind.log" python $TEST $BASEOPT
else
  echo python $TEST $BASEOPT
  python $TEST $BASEOPT
fi
