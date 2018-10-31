#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

DEVICE=$1
TEST_TO_RUN=$2
PE_CFG=$3
BITWIDTH=$4
IMGWIDTH=$5

if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

XDNN_SIZE=[3,${IMGWIDTH},${IMGWIDTH}]

images=`ls ${MLSUITE_ROOT}/xfdnn/tools/quantize/calibration_directory/*`
echo "Running with images: $images"

XCLBIN=xdnn_v2_32x56_2pe_16b_6mb_bank21.xclbin
if [ "$BITWIDTH" == "8" ]; then
  XCLBIN=xdnn_v2_32x56_2pe_8b_6mb_bank21.xclbin
fi

# Set Enviornment Variables corresponding to HW platform
. ${MLSUITE_ROOT}/overlaybins/setup.sh $DEVICE

# Build Non-Max Suppression C-code
cd nms
make 
cd ..

echo "=============== pyXDNN ============================="

if [ "$TEST_TO_RUN" == "e2e" ]; then
  #################
  # End To End, by default this will run 608x608, 16b quantization. 
  # Modify configs.py if you want to run a different end to end
  #################

  python yolo.py $DEVICE

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
#
#  python xyolo.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg yolo.cmds --datadir yolo.caffemodel_data --labels coco.names --xlnxlib $LIBXDNN_PATH --quantizecfg yolo_deploy_${IMGWIDTH}.json --firstfpgalayer conv0 --in_shape $XDNN_SIZE --images $images --style yolo
#
else
  echo "Hello, goodbye!"
fi
