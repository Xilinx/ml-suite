#!/usr/bin/env bash

DEVICE=$1

SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")

if [ -z "$MLSUITE_ROOT" ]; then
  export MLSUITE_ROOT=`python ${SCRIPT_DIR}/scripts/findRoot.py ${SCRIPT_DIR}`
fi

export LD_LIBRARY_PATH=${MLSUITE_ROOT}/overlaybins/${DEVICE}/runtime/lib/x86_64/:${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/build/lib:${MLSUITE_ROOT}/xfdnn/rt/lib:${MLSUITE_ROOT}/ext/boost/lib:$PWD

export XILINX_OPENCL=${MLSUITE_ROOT}/overlaybins/${DEVICE}
export LIBXDNN_PATH=${MLSUITE_ROOT}/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
export XCLBIN_PATH=${MLSUITE_ROOT}/overlaybins/${DEVICE}
export PYTHONPATH=${MLSUITE_ROOT}:${MLSUITE_ROOT}/apps/yolo:${MLSUITE_ROOT}/apps/yolo/nms:${MLSUITE_ROOT}/xfdnn/rt:${MLSUITE_ROOT}/xfdnn/tools/emu:${MLSUITE_ROOT}/xfdnn/tools/compile/network:${MLSUITE_ROOT}/xfdnn/tools/compile/graph:${MLSUITE_ROOT}/xfdnn/tools/compile/optimizations:${MLSUITE_ROOT}/xfdnn/tools/compile/codegeneration:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/version:${MLSUITE_ROOT}/xfdnn/tools/compile/memory:${MLSUITE_ROOT}/xfdnn/tools/compile/weights:${MLSUITE_ROOT}/xfdnn/tools/compile/bin:${MLSUITE_ROOT}/gemx/python
export SDACCEL_INI_PATH=${MLSUITE_ROOT}/overlaybins
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

#export XBLAS_EMIT_PROFILING_INFO=1
