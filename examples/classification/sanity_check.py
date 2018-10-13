#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

# Test all valid combinations of tests for alveo-u200
from subprocess import call

model_list    = ["googlenet_v1", "resnet50"]
bitwidth_list = [8, 16]
kcfg_list     = ["med", "large"]
nPE = {"med":4,"large":2}

test_enable = {"test_classify":True,"batch_classify":True,"streaming_classify":True,"multinet":True}

# Quantization for 16b ResNet50 is invalid (Need to fix)
# Run all configs targeting each legal PE individually

if test_enable["test_classify"]:
  for model in model_list:
    for bitwidth in bitwidth_list:
      for kcfg in kcfg_list:
        for pe in range(4):
          if pe >= nPE[kcfg]: # don't test PEs that don't exist
            continue
          else:
            call(["./run.sh","-p","alveo-u200","-t","test_classify","-k",kcfg,"-b",str(bitwidth),"-a",str(pe),"-m",model])

if test_enable["batch_classify"]:
  for model in model_list:
    for bitwidth in bitwidth_list:
      for kcfg in kcfg_list:
        batch_size = str(16*nPE[kcfg]/bitwidth)      
        call(["./run.sh","-p","alveo-u200","-t","batch_classify","-k",kcfg,"-b",str(bitwidth),"-s",batch_size,"-m",model])

# Coming soon ...
#if test_enable["streaming_classify"]:
#  for model in model_list:
#    for bitwidth in bitwidth_list:
#      for kcfg in kcfg_list:
#        batch_size = str(16*nPE[kcfg]/bitwidth)      
#        call(["./run.sh","-p","alveo-u200","-t","streaming_classify","-k",kcfg,"-b",str(bitwidth),"-s",batch_size,"-m",model])

# JSON FILE restricts parameter sweeping, only 1 config can be used
if test_enable["multinet"]:
  call(["./run.sh","-p","alveo-u200","-t","multinet","-k",kcfg_list[0],"-b",str(bitwidth_list[1])])
