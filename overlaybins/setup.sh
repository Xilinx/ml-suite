#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

MLSUITE_PLATFORM=$1

# In the future take from args?
# Need to figure out v2 and v3 support
XDNN_VER="2"

export MLSUITE_ROOT="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )/.." )"

echo "------------------"
echo "Using MLSUITE_ROOT"
echo "------------------"
echo $MLSUITE_ROOT
echo ""

# Initialize LD_LIBRARY_PATH
if [ -f /opt/xilinx/xrt/setup.sh ]; then
  . /opt/xilinx/xrt/setup.sh
else
  echo "--------------------------------------"
  echo "Skip sourcing /opt/xilinx/xrt/setup.sh"
  echo "--------------------------------------"
fi

RT_VER="20182"
LIBXDNN="libxfdnn.so"
if [ -z "$XILINX_XRT" ]; then
  RT_VER=20174
  LIBXDNN="libxfdnn.so.2017.4"
fi

GOLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLSUITE_ROOT}/ext/boost/lib:${MLSUITE_ROOT}/ext/boost/libs:${MLSUITE_ROOT}/ext/zmq/libs:${MLSUITE_ROOT}/ext/hdf5/lib
export LD_LIBRARY_PATH=${GOLD_LD_LIBRARY_PATH}:${MLSUITE_ROOT}/ext/sdx_build/runtime/lib/x86_64

echo "---------------------"
echo "Using LD_LIBRARY_PATH"
echo "---------------------"
echo $LD_LIBRARY_PATH

# This library is directly passed to Python
# First check if we have a built version
# Else default to prebuilt library
if [ -e ${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so ]; then
  export LIBXDNN_DIR=${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib
else
  export LIBXDNN_DIR=${MLSUITE_ROOT}/xfdnn/rt/libs
fi
  
export LIBXDNN_PATH=${LIBXDNN_DIR}/$LIBXDNN

echo "-------------------"
echo "Using LIBXDNN_PATH"
echo "-------------------"
echo $LIBXDNN_PATH
echo ""

# export PYTHONPATH=${MLSUITE_ROOT}:${MLSUITE_ROOT}/xfdnn/rt:${MLSUITE_ROOT}/ext:${MLSUITE_ROOT}/apps/yolo:${MLSUITE_ROOT}/apps/yolo/nms:${MLSUITE_ROOT}/xfdnn/tools/emu:${MLSUITE_ROOT}/xfdnn/tools/compile/network:${MLSUITE_ROOT}/xfdnn/tools/compile/graph:${MLSUITE_ROOT}/xfdnn/tools/compile/optimizations:${MLSUITE_ROOT}/xfdnn/tools/compile/codegeneration:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/version:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/weights:${MLSUITE_ROOT}/xfdnn/tools/compile/bin:${MLSUITE_ROOT}/xfdnn/tools/compile/parallel:${MLSUITE_ROOT}/xfdnn/tools/compile/pickle:${MLSUITE_ROOT}/xfmlp/python:${MLSUITE_ROOT}/xfdnn/rt/scripts/framework/caffe:${MLSUITE_ROOT}/xfdnn/rt/scripts/framework/darknet:${MLSUITE_ROOT}/xfdnn/rt/scripts/framework/base:${PYTHONPATH}

export PYTHONPATH=${MLSUITE_ROOT}:${MLSUITE_ROOT}/apps/yolo:${MLSUITE_ROOT}/apps/yolo/nms:${MLSUITE_ROOT}/xfmlp/python:${PYTHONPATH}

echo "-------------------"
echo "PYTHONPATH"
echo "-------------------"
echo $PYTHONPATH
echo ""

# Vince doesn't want the below to be default, use for debug only
# export SDACCEL_INI_PATH=${MLSUITE_ROOT}/overlaybins
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XBLAS_NUM_PREP_THREADS=4

if [ -z "$MLSUITE_PLATFORM" ]; then
  export MLSUITE_PLATFORM="(unknown)"
  AUTODETECT_PLATFORM=`python -c "import xfdnn.rt.xdnn as xdnn; print(xdnn.getHostDeviceName(None).decode('utf-8'))" | tr -d '\n'`
  if [ ! -z "$AUTODETECT_PLATFORM" -a $? -eq 0 -a `echo "$AUTODETECT_PLATFORM" | wc -w` -eq "1" ]; then
      #echo "Auto-detected platform: ${AUTODETECT_PLATFORM}"
      export MLSUITE_PLATFORM=${AUTODETECT_PLATFORM}
  else
    if [ -f /sys/hypervisor/uuid ] && [ `head -c 3 /sys/hypervisor/uuid` == ec2 ]; then
      export MLSUITE_PLATFORM=aws
    else
      echo "Warning: failed to auto-detect platform. Please manually specify platform with -p"
    fi
  fi
else
  export MLSUITE_PLATFORM=$MLSUITE_PLATFORM
fi

echo "-------------------"
echo "Using MLSUITE_PLATFORM"
echo "-------------------"
echo $MLSUITE_PLATFORM
echo ""

if [ -d "${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}" ]; then
  export XILINX_OPENCL=${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}
  export XCLBIN_PATH=${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}
  export LD_LIBRARY_PATH=${GOLD_LD_LIBRARY_PATH}:${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}/runtime/lib/x86_64
else
  echo "Warning: platform ${MLSUITE_PLATFORM} not supported"
fi

# Build NMS for YOLOv2 Demos
#make -C ${MLSUITE_ROOT}/apps/yolo/nms

#export XBLAS_EMIT_PROFILING_INFO=1
