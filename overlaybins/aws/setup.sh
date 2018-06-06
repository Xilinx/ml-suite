#!/usr/bin/env bash


SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
export ROOT=`python ${SCRIPT_DIR}/../scripts/findRoot.py`

export LD_LIBRARY_PATH=$ROOT/overlaybins/aws/runtime/lib/x86_64/:$ROOT/xfdnn/rt/xdnn_cpp/build/lib:$ROOT/xfdnn/rt/lib:$ROOT/ext/boost/lib:$PWD
export XILINX_OPENCL=$ROOT/overlaybins/aws
export LIBXDNN_PATH=$ROOT/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
export XCLBIN_PATH=$ROOT/overlaybins/aws
export PYTHONPATH=$ROOT:$ROOT/xfdnn/rt:$ROOT/xfdnn/tools/emu
export SDACCEL_INI_PATH=$ROOT/overlaybins
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

#export XBLAS_EMIT_PROFILING_INFO=1

#PE_CFG=$1
#if [ "$PE_CFG" == "med" ]; then
#  # 4 PE:
#  export XDNN_DDR_BANK=2,2,1,1
#  export XDNN_CSR_BASE=0x1800000,0x1800000,0x1810000,0x1810000
#  export XDNN_SLR_IDX=0,1,0,1
#elif [ "$PE_CFG" == "large" ]; then
#  # 2 PE:
#  export XDNN_DDR_BANK=2,1
#  export XDNN_CSR_BASE=0x1800000,0x1810000
#  export XDNN_SLR_IDX=0,0
#fi
