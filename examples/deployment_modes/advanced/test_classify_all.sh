#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/bin/bash

echo bvlc_googlenet_without_lrn
sh test_classify_bvlc_googlenet_without_lrn.sh
#echo deephi
#sh test_classify_deephi.sh
#echo resnet
#sh test_classify_resnet.sh
#echo squeezenet
#sh test_classify_squeezenet.sh
#echo vgg16
#sh test_classify_vgg16.sh
#echo mobilenet
#sh test_classify_mobilenet.sh
#echo inception_v3 
#sh test_classify_inception_v3.sh
echo flowers102
sh test_classify_flowers102.sh
echo places365
sh test_classify_places365.sh
#echo aiotlabs
#sh test_classify_aiotlabs.sh

