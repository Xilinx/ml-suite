#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "./run.sh -p <platform> -t e2e"
}

# Parse Options
OPTS=`getopt -o p:t:b:vh --long platform:,test:,bitwidth:,verbose,help -n "$0" -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi
while true
do
  case "$1" in
    -p|--platform    ) MLSUITE_PLATFORM="$2" ; shift 2 ;;
    -b|--bitwidth    ) BITWIDTH="$2"         ; shift 2 ;;
    -t|--test        ) TEST_TO_RUN="$2"      ; shift 2 ;;
    -v|--verbose     ) VERBOSE=1             ; shift 1 ;;
    -h|--help        ) usage                 ; exit  1 ;;
     *) break ;;
  esac
done

if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=$PWD/../..
fi

if [ -z $TEST_TO_RUN ]; then
   echo "Setting test to 'e2e'"
   TEST_TO_RUN="e2e"
fi

images=`ls ${MLSUITE_ROOT}/xfdnn/tools/quantize/calibration_directory/*`
echo "Running with images: $images"


# Set Environment Variables corresponding to HW platform
if [ -z $MLSUITE_PLATFORM ]; then
   echo "Platform not set.  Autodetecting..."
fi
. ${MLSUITE_ROOT}/overlaybins/setup.sh ${MLSUITE_PLATFORM}

# Build Non-Max Suppression C-code
cd nms
make 
cd ..

echo "=============== pyXDNN ============================="
echo "Platform: $MLSUITE_PLATFORM"
echo "Test: $TEST_TO_RUN"

if [ "$TEST_TO_RUN" == "e2e" ]; then
  #################
  # End To End, by default this will run 608x608, 16b quantization. 
  # Modify configs.py if you want to run a different end to end
  #################

  python yolo.py ${MLSUITE_PLATFORM}

## Note: The xyolo module was written to support being called directly at the CLI
##       yolo.py supports running the offline steps (Compile,Quantize, and bringing in xyolo as a class object)
##       However, xyolo can be called from CLI with command line arguments
##       The code below shows an example of how this can be done, but it won't
##       work unless you have already ran the Compile,Quantize steps, and
##       saved the results locally
#elif [ "$TEST_TO_RUN" == "deploy" ]; then
#  ############################
#  # Test single deploy, you must have previously compiled, and quantized
#  ############################
#IMGWIDTH=608
#XDNN_SIZE=[3,${IMGWIDTH},${IMGWIDTH}]
#
#XCLBIN=xdnn_v2_32x56_2pe_16b_6mb_bank21.xclbin
#if [ "$BITWIDTH" == "8" ]; then
#  XCLBIN=xdnn_v2_32x56_2pe_8b_6mb_bank21.xclbin
#fi
#
#  python xyolo.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg yolo.cmds --datadir yolo.caffemodel_data --labels coco.names --xlnxlib $LIBXDNN_PATH --quantizecfg yolo_deploy_${IMGWIDTH}.json --firstfpgalayer conv0 --in_shape $XDNN_SIZE --images $images --style yolo
#
else
  usage
fi
