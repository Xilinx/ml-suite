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

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

if [ -z "$MLSUITE_ROOT" ]; then
  export MLSUITE_ROOT=`python ${SCRIPT_DIR}/scripts/findRoot.py ${SCRIPT_DIR}`
else
  export MLSUITE_ROOT=$MLSUITE_ROOT
fi
  
echo "------------------"
echo "Using MLSUITE_ROOT"
echo "------------------"
echo $MLSUITE_ROOT
echo "------------------"

# Initialize LD_LIBRARY_PATH

if [ -f /opt/xilinx/xrt/setup.sh ]; then
  . /opt/xilinx/xrt/setup.sh
else
  echo "--------------------------------------"
  echo "Skip sourcing /opt/xilinx/xrt/setup.sh"
  echo "--------------------------------------"
fi

RT_VER=20182
if [ -z "$XILINX_XRT" ]; then
  RT_VER=20174
fi

GOLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MLSUITE_ROOT}/ext/boost/lib:${MLSUITE_ROOT}/ext/zmq/libs
export LD_LIBRARY_PATH=${GOLD_LD_LIBRARY_PATH}:${MLSUITE_ROOT}/ext/sdx_build/runtime/lib/x86_64

# This library is directly passed to Python
# First check if we have a built version
# Else default to prebuilt library
if [ -e ${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so ]; then
  export LIBXDNN_DIR=${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib
  export LIBXDNN_PATH=${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
else
  export LIBXDNN_DIR=${MLSUITE_ROOT}/xfdnn/rt/libs
  export LIBXDNN_PATH=${LIBXDNN_DIR}/libxfdnn.so.${XDNN_VER}.${RT_VER}
fi

echo "-------------------"
echo "Using LIBXDNN_PATH"
echo "-------------------"
echo $LIBXDNN_PATH
echo "-------------------"

export PYTHONPATH=${MLSUITE_ROOT}:${MLSUITE_ROOT}/xfdnn/rt:${MLSUITE_ROOT}/ext:${MLSUITE_ROOT}/models/darknet/tools:${MLSUITE_ROOT}/apps/yolo:${MLSUITE_ROOT}/apps/yolo/nms:${MLSUITE_ROOT}/xfdnn/tools/emu:${MLSUITE_ROOT}/xfdnn/tools/compile/network:${MLSUITE_ROOT}/xfdnn/tools/compile/graph:${MLSUITE_ROOT}/xfdnn/tools/compile/optimizations:${MLSUITE_ROOT}/xfdnn/tools/compile/codegeneration:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/version:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/weights:${MLSUITE_ROOT}/xfdnn/tools/compile/bin:${MLSUITE_ROOT}/xfdnn/tools/compile/parallel:${MLSUITE_ROOT}/xfmlp/python
export SDACCEL_INI_PATH=${MLSUITE_ROOT}/overlaybins
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export XBLAS_NUM_PREP_THREADS=4

if [ -z "$MLSUITE_PLATFORM" ]; then
  export MLSUITE_PLATFORM="(unknown)"
  AUTODETECT_PLATFORM=`python -c "import xdnn; print xdnn.getHostDeviceName(None)" | tr -d '\n'`
  if [ $? -eq 0 -a ! -z $AUTODETECT_PLATFORM ]; then
    echo "Auto-detected platform: ${AUTODETECT_PLATFORM}"
    export MLSUITE_PLATFORM=${AUTODETECT_PLATFORM}
  else
    echo "Warning: failed to auto-detect platform. Please manually specify platform with -p"
  fi
fi

if [ -d "${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}" ]; then
:
else
echo "Warning: platform ${MLSUITE_PLATFORM} not supported"
fi

export XILINX_OPENCL=${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}
export XCLBIN_PATH=${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}
export LD_LIBRARY_PATH=${GOLD_LD_LIBRARY_PATH}:${MLSUITE_ROOT}/overlaybins/${MLSUITE_PLATFORM}/runtime/lib/x86_64

# Build NMS for YOLOv2 Demos
#make -C ${MLSUITE_ROOT}/apps/yolo/nms

#export XBLAS_EMIT_PROFILING_INFO=1
