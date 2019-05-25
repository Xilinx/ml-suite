#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import os,sys

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler

def run_compiler(dsp, mem, prototxt, caffemodel):

  print("Testing xfdnn_compiler_caffe...")

  print("Testing:\n  prototxt %s\n  caffemodel %s\n  dsp %s\n  mem %s\n" % (prototxt,caffemodel,dsp,mem))
  compiler = xfdnnCompiler(
      networkfile=prototxt,
      weights=caffemodel,
      dsp=dsp,
      memory=mem,
      generatefile="work/"+prototxt.replace('/','_')+"_"+str(dsp)+"/fpga.cmds",
      anew="work/"+prototxt.replace('/','_')+"_"+str(dsp)+"/optimized_model"
      )
  SUCCESS = compiler.compile()
  #assert(SUCCESS)
  # Compiler will throw exception if it does not succeed as of 3/12/19
  del compiler


def get_caffe_model_list_all():

  import os
  #print(os.getcwd()))

  if "MLSUITE_ROOT" in os.environ:
    path_prepend = os.environ["MLSUITE_ROOT"] + "/"
  else:
    path_prepend = ""


  prototxt_list = [ \
    path_prepend + "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt", \
    path_prepend + "models/caffe/inception_v3/fp32/inception_v3_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/aiotlabs/fp32/resnet18_baseline_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/aiotlabs/fp32/resnet18_emdnn_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/deephi/fp32/resnet50_pruned_deploy.prototxt", \
    path_prepend + "models/caffe/resnet/fp32/resnet50_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/resnet/fp32/resnet101_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/resnet/fp32/resnet152_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/squeezenet/fp32/squeezenet_deploy.prototxt", \
    path_prepend + "models/caffe/mobilenet/fp32/mobilenet_without_bn_deploy.prototxt", \
    path_prepend + "models/caffe/vgg16/fp32/vgg16_deploy.prototxt", \
    path_prepend + "models/caffe/yolov2/fp32/yolo_deploy_224.prototxt", \
    path_prepend + "models/caffe/yolov2/fp32/yolo_deploy_416.prototxt", \
    path_prepend + "models/caffe/yolov2/fp32/yolo_deploy_608.prototxt", \
#    "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes/license_plate_recognition_quantizations.prototxt", \
#    /wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes/car_logo_recognition_quantizations.prototxt", \
#    /wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes/car_attributes_recognition_quantizations.prototxt", \
#    /wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes/pedestrian_attributes_recognition_quantizations.prototxt", \
#    /wrk/acceleration/models/deephi/reid_model_release_20190301/deploy.prototxt", \
#    /wrk/acceleration/models/deephi/Car_Logo_Detection/deploy.prototxt", \
#    /wrk/acceleration/models/deephi/Plate_Detection/deploy.prototxt", \
#    /wrk/acceleration/models/deephi/Pedestrian_Detection_INT8_models_test_codes/deploy.prototxt", \
  ]

  caffemodel_list = [ \
    path_prepend + "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel", \
    path_prepend + "models/caffe/inception_v3/fp32/inception_v3_without_bn.caffemodel", \
    path_prepend + "models/caffe/aiotlabs/fp32/resnet18_baseline_without_bn.caffemodel", \
    path_prepend + "models/caffe/aiotlabs/fp32/resnet18_emdnn_without_bn.caffemodel", \
    path_prepend + "models/caffe/deephi/fp32/resnet50_pruned.caffemodel", \
    path_prepend + "models/caffe/resnet/int8/resnet50_without_bn_quantized.caffemodel", \
    path_prepend + "models/caffe/resnet/fp32/resnet101_without_bn.caffemodel", \
    path_prepend + "models/caffe/resnet/fp32/resnet152_without_bn.caffemodel", \
    path_prepend + "models/caffe/squeezenet/fp32/squeezenet.caffemodel", \
    path_prepend + "models/caffe/mobilenet/fp32/mobilenet_without_bn.caffemodel", \
    path_prepend + "models/caffe/vgg16/fp32/vgg16.caffemodel", \
    path_prepend + "models/caffe/yolov2/fp32/yolov2.caffemodel", \
    path_prepend + "models/caffe/yolov2/fp32/yolov2.caffemodel", \
    path_prepend + "models/caffe/yolov2/fp32/yolov2.caffemodel", \
#    "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes/license_plate_recognition_quantizations.caffemodel", \
#    "/wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes/car_logo_recognition_quantizations.caffemodel", \
#    "/wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes/car_attributes_recognition_quantizations.caffemodel", \
#    "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes/pedestrian_attributes_recognition_quantizations.caffemodel", \
#    "/wrk/acceleration/models/deephi/reid_model_release_20190301/deploy.caffemodel", \
#    "/wrk/acceleration/models/deephi/Car_Logo_Detection/deploy.caffemodel", \
#    "/wrk/acceleration/models/deephi/Plate_Detection/deploy.caffemodel", \
##    "/wrk/acceleration/models/deephi/Pedestrian_Detection_INT8_models_test_codes/deploy.caffemodel", \
  ]


  return (prototxt_list,caffemodel_list)





def get_caffe_model_list_1():

  prototxt_list = [ \
    "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt" \
  ]

  caffemodel_list = [ \
    "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel" \
  ]

  return (prototxt_list,caffemodel_list)



def get_caffe_model_list():
  (p,c) = get_caffe_model_list_all()
  return list(zip(p,c))

testdata = get_caffe_model_list()


import pytest

@pytest.mark.parametrize('prototxt,caffemodel', testdata)
def test_xfdnn_compiler_caffe_28(prototxt,caffemodel):
  print("Testing xfdnn_compiler_caffe...")

  dsp = 28
  mem = 4

  run_compiler(dsp,mem,prototxt,caffemodel)

@pytest.mark.parametrize('prototxt,caffemodel', testdata)
def test_xfdnn_compiler_caffe_56(prototxt,caffemodel):
  print("Testing xfdnn_compiler_caffe...")

  dsp = 56
  mem = 6

  run_compiler(dsp,mem,prototxt,caffemodel)


@pytest.mark.parametrize('prototxt,caffemodel', testdata)
def test_xfdnn_compiler_caffe_96(prototxt,caffemodel):
  print("Testing xfdnn_compiler_caffe...")

  dsp = 96
  mem = 9

  run_compiler(dsp,mem,prototxt,caffemodel)


if __name__ == "__main__":
  for (prototxt,caffemodel) in testdata:
	test_xfdnn_compiler_caffe_28(prototxt,caffemodel)
        test_xfdnn_compiler_caffe_56(prototxt,caffemodel)
        test_xfdnn_compiler_caffe_96(prototxt,caffemodel)


