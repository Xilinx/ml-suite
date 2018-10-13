#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

DEVICE=$1

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

export MLSUITE_ROOT=`python ${SCRIPT_DIR}/scripts/findRoot.py ${SCRIPT_DIR}`

#When we fully switch to 2018.2+, we should force the user to set this var, and exit if it isn't set
#export XILINX_XRT=/opt/xilinx/xrt

export LD_LIBRARY_PATH=${MLSUITE_ROOT}/overlaybins/${DEVICE}/runtime/lib/x86_64/:${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/build/lib:${MLSUITE_ROOT}/xfdnn/rt/lib:${MLSUITE_ROOT}/ext/boost/lib:${MLSUITE_ROOT}/ext/zmq/libs:$PWD

. /opt/xilinx/xrt/setup.sh || true

# If the above script exists it will be ran, and XILINX_XRT will be set

export XILINX_OPENCL=${MLSUITE_ROOT}/overlaybins/${DEVICE}
export LIBXDNN_DIR=${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib
export LIBXDNN_PATH=${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
export XCLBIN_PATH=${MLSUITE_ROOT}/overlaybins/${DEVICE}
export PYTHONPATH=${MLSUITE_ROOT}:${MLSUITE_ROOT}/xfdnn/rt:${MLSUITE_ROOT}/ext:${MLSUITE_ROOT}/apps/yolo:${MLSUITE_ROOT}/apps/yolo/nms:${MLSUITE_ROOT}/xfdnn/tools/emu:${MLSUITE_ROOT}/xfdnn/tools/compile/network:${MLSUITE_ROOT}/xfdnn/tools/compile/graph:${MLSUITE_ROOT}/xfdnn/tools/compile/optimizations:${MLSUITE_ROOT}/xfdnn/tools/compile/codegeneration:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/version:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/weights:${MLSUITE_ROOT}/xfdnn/tools/compile/bin:${MLSUITE_ROOT}/xfdnn/tools/compile/parallel:${MLSUITE_ROOT}/xfmlp/python
export SDACCEL_INI_PATH=${MLSUITE_ROOT}/overlaybins
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

export XDNN_VER=2

export RT_VER=20182
if [ -z "$XILINX_XRT" ]; then
  export RT_VER=20174
fi

# Need to soft link to real binary to support multiple version of XRT
ln -sf ${LIBXDNN_DIR}/libxfdnn.so.${XDNN_VER}.${RT_VER} $LIBXDNN_PATH 

echo $RT_VER

# Build NMS for YOLOv2 Demos
make -C ${MLSUITE_ROOT}/apps/yolo/nms

#export XBLAS_EMIT_PROFILING_INFO=1
