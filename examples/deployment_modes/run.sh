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
  echo "<test>     : test_classify / streaming_classify / streaming_classify_benchmark"
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
####################
# single image test
####################
if [ "$TEST" == "test_classify" ]; then
  TEST=test_classify.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg

  fi
  BASEOPT+=" --images $DIRECTORY"
####################
# network profile
####################
elif [ "$TEST" == "profile" ]; then
  TEST=profile.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg

  fi
  BASEOPT+=" --images $DIRECTORY"
############################
# multi-process streaming
############################
elif [[ "$TEST" == "streaming_classify"* ]]; then
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi

  if [ "$TEST" == "streaming_classify_benchmark" ]; then
    BASEOPT+=" --benchmarkmode 1"
  fi 
  BASEOPT+=" --numstream $NUMSTREAMS"
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --numprepproc $NUMPREPPROC"
  if [ "$PERPETUAL" == 1 ]; then
    BASEOPT+=" --zmqpub --perpetual --deviceID $DEVICEID"
  fi

  TEST=mp_classify.py

###########################
# switch to run the classification examples through c++ APIs
# runs with 8 bit quantization for now
###########################

elif [ "$TEST" == "classify_cpp" ]; then
  cd classify_cpp
  make
  cp ./classify.exe ../classify.exe
  cd -
  if [ "$MODEL" == "googlenet_v1" ]; then
     FPGAOUTSZ=1024
  elif [ "$MODEL" == "resnet50" ];then
     FPGAOUTSZ=2048
  fi
  BATCHSIZE=2
  DIRECTORY=$MLSUITE_ROOT/examples/deployment_modes/dog.jpg
  BASEOPT_CPP="--xclbin $XCLBIN_PATH/$XCLBIN --netcfg $NETCFG --fpgaoutsz $FPGAOUTSZ --datadir $WEIGHTS --labels ./synset_words.txt --quantizecfg $QUANTCFG --img_input_scale $IMG_INPUT_SCALE --batch_sz $BATCHSIZE"
  BASEOPT_CPP+=" --image $DIRECTORY"
  BASEOPT_CPP+=" --in_w 224 --in_h 224 --out_w 1 --out_h 1 --out_d $FPGAOUTSZ"

  OPENCV_LIB=${MLSUITE_ROOT}/opencv_lib
  HDF5_PATH=${MLSUITE_ROOT}/ext/hdf5
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MLSUITE_ROOT/ext/zmq/libs:$MLSUITE_ROOT/ext/boost/libs:$MLSUITE_ROOT/ext/sdx_build/runtime/lib/x86_64:${HDF5_PATH}/lib:$OPENCV_LIB
  #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MLSUITE_ROOT/xfdnn/rt/xdnn_cpp/lib:$MLSUITE_ROOT/ext/zmq/libs:$MLSUITE_ROOT/ext/boost/libs:$MLSUITE_ROOT/ext/sdx_build/runtime/lib/x86_64:${HDF5_PATH}/lib:$OPENCV_LIB
  if [ "$KCFG" == "v3" ]; then
	  cp $MLSUITE_ROOT/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so.v3 $OPENCV_LIB/libxfdnn.so
  else
	  cp $MLSUITE_ROOT/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so $OPENCV_LIB/libxfdnn.so
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

