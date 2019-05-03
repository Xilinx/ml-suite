#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
gcc -c -fPIC nms.c -o nms.o
#g++ -c -fPIC nms.c -o nms.o
g++ -shared -Wl,-soname,libnms.so -o libnms.so nms.o
