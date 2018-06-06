#!/bin/bash

export XILINX_SDACCEL=../../../../ext/sdx_build

g++ -g -std=c++11 -D __USE_XOPEN2K8 -I ${XILINX_SDACCEL}/include -I ${XILINX_SDACCEL}/runtime/include/1_2 -I ../ -L ../lib -L ${XILINX_SDACCEL}/runtime/lib/x86_64 main.cpp -o test.exe -lxfdnn -lboost_thread -lxilinxopencl
