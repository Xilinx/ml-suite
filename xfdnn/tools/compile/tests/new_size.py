#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from __future__ import print_function
import argparse
import sys
import os
import copy
import numpy

Directories = ["codegeneration","graph","memory","network","optimizations","quantization","version", "weights","tests"]


## XFDNN_ROOT
XFDNN = os.environ['XFDNN_ROOT'] if 'XFDNN_ROOT' in os.environ else None

if not XFDNN:
    print("WARNING: Environment XFDNN_ROOT not set.  It should point to parent directory of xfdnn_tools")
    cwd = os.getcwd()
else:
    print("INFO: Environment XFDNN_ROOT set to",XFDNN)
    cwd = XFDNN + "/xfdnn_tools/compile"
    
for d in Directories:
    sys.path.insert(0, os.path.abspath(cwd+"/"+d))

import dagtools_type


if __name__ == "__main__":   

    import dagtools_type
    import hardware
    import messages

    
    messages.DEBUG(True)
    print(hardware.hw_abstraction.to_string())
    shape = dagtools_type.SizeType(batches=1, channels=256, height=28, width=28)
    small_shape = dagtools_type.SizeType(1,32,24,24)
    print("shape",shape)
    
    
    
    print("V3 AT ", hardware.hw_abstraction_complex.slice_bytes_and_time(shape))
    print("V3 DDR", hardware.hw_abstraction_complex.ddr_bytes_and_time(shape))
    print("V2 AT ",hardware.hw_abstraction_complex.slice_bytes_and_time(shape,slice=1))
    print("V2 DDR",hardware.hw_abstraction_complex.ddr_bytes_and_time(shape,slice=1))


    x =  dagtools_type.memory_allocation(0,  hardware.hw_abstraction_complex.slice_bytes_and_time(shape).space, shape)
    y =  dagtools_type.memory_allocation(0,  hardware.hw_abstraction_complex.slice_bytes_and_time(shape).space, shape)
    ys =  dagtools_type.memory_allocation(0,  hardware.hw_abstraction_complex.slice_bytes_and_time(shape).space, small_shape)

    repl = hardware.replication_default()
    repl = repl._replace(repl_unit_width=16) 
    x = x._replace(replication=repl)

    print("X size:",x)

    newx = hardware.hw_abstraction_complex.allocate(x)

    print("X allocated:", newx)

    newy = hardware.hw_abstraction_complex.allocate(y)

    print("Y allocated:", newy)
    ys = ys._replace(replication=repl)

    newys = hardware.hw_abstraction_complex.allocate(ys)

    print("YS allocated:", newys)


    hardware.hw_abstraction_complex.free(newy)
    hardware.hw_abstraction_complex.free(newx)

    print("SLICE 0")
    print("move up", newys)
    y1 = hardware.hw_abstraction_complex.move_up(newys)
    print("YS allocated:", y1)

    print("move_down", y1)
    newys = hardware.hw_abstraction_complex.move_down(y1)
    print("YS allocated:", newys)

    hardware.hw_abstraction_complex.free(newys)

    print("SLICE 1")
    repl = repl._replace(repl_unit_width=0)
    ys =  dagtools_type.memory_allocation(0,  hardware.hw_abstraction_complex.slice_bytes_and_time(shape).space, small_shape)
    ys = ys._replace(replication=repl,slice=1)

    newys = hardware.hw_abstraction_complex.allocate(ys)
    print("move up", newys)
    y1 = hardware.hw_abstraction_complex.move_up(newys)
    print("YS allocated:", y1)

    print("move_down", y1)
    newys = hardware.hw_abstraction_complex.move_down(y1)
    print("YS allocated:", newys)

    hardware.hw_abstraction_complex.free(newys)
    
