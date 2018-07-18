#!/usr/bin/env bash


SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
export ROOT=`python ${SCRIPT_DIR}/../scripts/findRoot.py ${SCRIPT_DIR}`

export LD_LIBRARY_PATH=$ROOT/overlaybins/1525/runtimev3/lib/x86_64/:$ROOT/xfdnn/rt/xdnn_cpp/build/lib:$ROOT/xfdnn/rt/lib:$ROOT/ext/boost/lib:$PWD
export XILINX_OPENCL=$ROOT/overlaybins/1525
export LIBXDNN_PATH=$ROOT/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
export XCLBIN_PATH=$ROOT/overlaybins/1525
export PYTHONPATH=$ROOT:$ROOT/xfdnn/rt:$ROOT/xfdnn/tools/emu:$ROOT/xfdnn/tools/compile/network:$ROOT/xfdnn/tools/compile/graph:$ROOT/xfdnn/tools/compile/optimizations:$ROOT/xfdnn/tools/compile/codegeneration:$ROOT/xfdnn/tools/compile/memory:$ROOT/xfdnn/tools/compile/version:$ROOT/xfdnn/tools/compile/memory:$ROOT/xfdnn/tools/compile/weights
export SDACCEL_INI_PATH=$ROOT/overlaybins
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

#
##!/usr/bin/env bash
#
#
#SCRIPT_DIR=$(dirname "${BASH_SOURCE[0]}")
#export ROOT=`python ${SCRIPT_DIR}/../scripts/findRoot.py`
#
#export LD_LIBRARY_PATH=$ROOT/overlaybins/1525/runtimev3/lib/x86_64/:$ROOT/xfdnn/rt/xdnn_cpp/build/lib:$ROOT/xfdnn/rt/lib:$ROOT/ext/boost/lib:$PWD
#export XILINX_OPENCL=$ROOT/overlaybins/1525
#export LIBXDNN_PATH=$ROOT/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so
#export XCLBIN_PATH=$ROOT/overlaybins/1525
#export PYTHONPATH=$ROOT:$ROOT/xfdnn/rt:$ROOT/xfdnn/tools/emu
#export SDACCEL_INI_PATH=$ROOT/overlaybins
#export OMP_NUM_THREADS=4
#export MKL_NUM_THREADS=4
