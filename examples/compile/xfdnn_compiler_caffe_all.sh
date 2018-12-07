#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/bin/bash

echo bvlc_googlenet_without_lrn
sh xfdnn_compiler_caffe_bvlc_googlenet_without_lrn.sh
echo deephi
sh xfdnn_compiler_caffe_deephi.sh
echo resnet
sh xfdnn_compiler_caffe_resnet.sh
echo squeezenet
sh xfdnn_compiler_caffe_squeezenet.sh
echo vgg16
sh xfdnn_compiler_caffe_vgg16.sh
echo mobilenet
sh xfdnn_compiler_caffe_mobilenet.sh
echo inception_v3
sh xfdnn_compiler_caffe_inception_v3.sh
echo flowers102
sh xfdnn_compiler_caffe_flowers102.sh
echo places365
sh xfdnn_compiler_caffe_places365.sh
echo aiotlabs
sh xfdnn_compiler_caffe_aiotlabs.sh

