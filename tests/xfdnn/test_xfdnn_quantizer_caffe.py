#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import os,sys

from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizer

def test_xfdnn_quantizer_caffe():

  print "Testing xfdnn_quantizer_caffe..."
  
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

  bitwidths_list = [8, 16]

  for prototxt, caffemodel in zip(prototxt_list,caffemodel_list):
    for bw in bitwidths_list:
      print("Testing:\n  prototxt %s\n  caffemodel %s\n  bitwidth %s" % (prototxt,caffemodel,bw))
      quantizer = xfdnnQuantizer(
        deploy_model=prototxt,
        weights=caffemodel,
        bitwidths=[bw,bw,bw],
        output_json="work/quantization_params.json",
        calibration_directory="./xfdnn/tools/quantize/calibration_directory",
        transpose=[2,0,1],                             
        channel_swap=[2,1,0],                          
        raw_scale=255.0,                                   
        mean_value=[104.0,117.0,123.0],
        input_scale=1.0                                
      )
      SUCCESS = quantizer.quantize()
      assert(SUCCESS)
      del quantizer

