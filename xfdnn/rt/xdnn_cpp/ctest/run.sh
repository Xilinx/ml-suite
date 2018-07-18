#!/bin/bash

export LD_LIBRARY_PATH=./runtime/lib/x86_64/:../lib
export XILINX_OPENCL=.
export SDACCEL_INI_PATH=$PWD

export XDNN_DDR_BANK=2
export XDNN_CSR_BASE="0x0"
export XDNN_SLR_IDX=0
#export XDNN_VERBOSE=1

export XDNN_GLOBAL_SCALE_A=10000
export XDNN_GLOBAL_SCALE_B=30

./test.exe
