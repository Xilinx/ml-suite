#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import os,sys

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler

def test_xfdnn_compiler_caffe():

  print "Testing xfdnn_compiler_caffe..."
  
  prototxt_list = [ \
    "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt", \
    "models/caffe/inception_v3/fp32/inception_v3_without_bn_deploy.prototxt", \
    "models/caffe/aiotlabs/fp32/resnet18_baseline_without_bn_deploy.prototxt", \
    "models/caffe/aiotlabs/fp32/resnet18_emdnn_without_bn_deploy.prototxt", \
    "models/caffe/deephi/fp32/resnet50_pruned_deploy.prototxt", \
    "models/caffe/resnet/fp32/resnet50_without_bn_deploy.prototxt", \
    "models/caffe/resnet/fp32/resnet101_without_bn_deploy.prototxt", \
    "models/caffe/resnet/fp32/resnet152_without_bn_deploy.prototxt", \
    "models/caffe/squeezenet/fp32/squeezenet_deploy.prototxt", \
    "models/caffe/mobilenet/fp32/mobilenet_without_bn_deploy.prototxt", \
    "models/caffe/vgg16/fp32/vgg16_deploy.prototxt", \
    "models/yolov2/caffe/fp32/yolo_deploy_224.prototxt", \
    "models/yolov2/caffe/fp32/yolo_deploy_416.prototxt", \
    "models/yolov2/caffe/fp32/yolo_deploy_608.prototxt", \
  ]

  caffemodel_list = [ \
    "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel", \
    "models/caffe/inception_v3/fp32/inception_v3_without_bn.caffemodel", \
    "models/caffe/aiotlabs/fp32/resnet18_baseline_without_bn.caffemodel", \
    "models/caffe/aiotlabs/fp32/resnet18_emdnn_without_bn.caffemodel", \
    "models/caffe/deephi/fp32/resnet50_pruned.caffemodel", \
    "models/caffe/resnet/int8/resnet50_without_bn_quantized.caffemodel", \
    "models/caffe/resnet/fp32/resnet101_without_bn.caffemodel", \
    "models/caffe/resnet/fp32/resnet152_without_bn.caffemodel", \
    "models/caffe/squeezenet/fp32/squeezenet.caffemodel", \
    "models/caffe/mobilenet/fp32/mobilenet_without_bn.caffemodel", \
    "models/caffe/vgg16/fp32/vgg16.caffemodel", \
    "models/yolov2/caffe/fp32/yolov2.caffemodel", \
    "models/yolov2/caffe/fp32/yolov2.caffemodel", \
    "models/yolov2/caffe/fp32/yolov2.caffemodel", \
  ]

  dsp_list = [28, 56, 96]
  mem_list = [4,   6,  9]

  for prototxt, caffemodel in zip(prototxt_list,caffemodel_list):
    for dsp,mem in zip(dsp_list,mem_list):
      print("Testing:\n  prototxt %s\n  caffemodel %s\n  dsp %s\n  mem %s\n" % (prototxt,caffemodel,dsp,mem))
      compiler = xfdnnCompiler(
        networkfile=prototxt,
        weights=caffemodel,
        dsp=dsp,
        memory=mem,
        generatefile="work/fpga.cmds",
        anew="work/optimized_model"
      )
      SUCCESS,_ = compiler.compile()
      assert(SUCCESS)
      del compiler

