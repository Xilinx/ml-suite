#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/bin/bash 
. ../../overlaybins/setup.sh 
XCLBIN=overlay_3.xclbin
DSP_WIDTH=56
BITWIDTH=16
DATAFILES_PATH=../../examples/classification

python serve.py --xclbin $XCLBIN_PATH/$XCLBIN --netcfg ${DATAFILES_PATH}/data/googlenet_v1_${DSP_WIDTH}.json --fpgaoutsz 1024 --datadir ${DATAFILES_PATH}/data/googlenet_v1_data --labels ${DATAFILES_PATH}/synset_words.txt --quantizecfg ${DATAFILES_PATH}/data/googlenet_v1_${BITWIDTH}b.json --images . 
