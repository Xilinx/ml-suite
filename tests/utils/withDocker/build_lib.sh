#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
#!/bin/bash
echo "### Cleaning Stale Files From Previous Run ###"
#rm -rf output_logs
#rm nw_status.txt

# Logs directory
#mkdir output_logs

# Select platform
#export PLATFORM=alveo-u200 
#export PLATFORM=1525 

# Export MLSuite path
export MLSUITE_ROOT=..

# pull latest 
git pull -r

# pull lfs files
export PATH=$PATH:/wrk/acceleration/MLsuite_Embedded/anup/gitlfs/
#git lfs pull

# Enable below to build xdnn lib on CentOS 7.4
#export PATH=/tools/batonroot/rodin/devkits/lnx64/binutils-2.26/bin:/tools/batonroot/rodin/devkits/lnx64/make-4.1/bin:/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/bin:$PATH
#export LD_LIBRARY_PATH=/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/lib64:$LD_LIBRARY_PATH

# Build rt
cd ../xfdnn/rt/xdnn_cpp
make clean;make -j8
cd -

#Build nms for yolo
cd $MLSUITE_ROOT/apps/yolo/nms
make
cd -

