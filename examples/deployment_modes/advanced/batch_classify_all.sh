#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/bin/bash

echo bvlc_googlenet_without_lrn
sh batch_classify_bvlc_googlenet_without_lrn.sh
echo deephi
sh batch_classify_deephi.sh
echo resnet
sh batch_classify_resnet.sh
#echo squeezenet
#sh batch_classify_squeezenet.sh
#echo vgg16
#sh batch_classify_vgg16.sh
echo mobilenet
sh batch_classify_mobilenet.sh
#echo inception_v3 
#sh batch_classify_inception_v3.sh
#echo flowers102
#sh batch_classify_flowers102.sh
#echo places365
#sh batch_classify_places365.sh
#echo aiotlabs
#sh batch_classify_aiotlabs.sh

