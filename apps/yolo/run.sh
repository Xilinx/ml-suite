#!/usr/bin/env bash

DEVICE=$1
TEST_TO_RUN=$2
PE_CFG=$3
BITWIDTH=$4
IMGWIDTH=$5

XDNN_SIZE=[3,${IMGWIDTH},${IMGWIDTH}]

images=`ls calibration_directory/*`
echo "Running with images: $images"

if [ "$PE_CFG" == "med" ]; then
  DSP_WIDTH=28
else
  DSP_WIDTH=56
fi

# Set Enviornment Variables corresponding to HW platform
. ../../overlaybins/$DEVICE/setup.sh

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

  python yolo.py

elif [ "$TEST_TO_RUN" == "deploy" ]; then
  ############################
  # Test single deploy, you must have previously compiled, and quantized
  ############################

  python xyolo.py --xclbin $XCLBIN_PATH/xdnn_${DSP_WIDTH}_${BITWIDTH}b_5m.xclbin --netcfg yolo.cmds --datadir yolo.caffemodel_data --labels coco.names --xlnxlib $LIBXDNN_PATH --quantizecfg yolo_deploy_${IMGWIDTH}.json --firstfpgalayer conv0 --in_shape $XDNN_SIZE --images $images --style yolo

else
  echo "Hello, goodbye!"
fi
