#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import os,sys

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler

def run_compiler(dsp, mem, prototxt, caffemodel, quantcfg):

  print("Testing xfdnn_compiler_caffe...")

  print("Testing:\n  prototxt %s\n  caffemodel %s\n quantization file %s\n dsp %s\n  mem %s\n" % (prototxt,caffemodel,quantcfg,dsp,mem))
  compiler = xfdnnCompiler(
      usedeephi=True,
      cpulayermustgo=True,
      pipelineconvmaxpool=True,
      quant_cfgfile=quantcfg,
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

  prototxt_list = [ \
    "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes/license_plate_recognition_quantizations.prototxt", \
    "/wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes/car_logo_recognition_quantizations.prototxt", \
    "/wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes/car_attributes_recognition_quantizations.prototxt", \
    "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes/pedestrian_attributes_recognition_quantizations.prototxt", \
    "/wrk/acceleration/models/deephi/reid_model_release_20190301/deploy.prototxt", \
    "/wrk/acceleration/models/deephi/Car_Logo_Detection/deploy.prototxt", \
    "/wrk/acceleration/models/deephi/Plate_Detection/deploy.prototxt", \
#    "/wrk/acceleration/models/deephi/Pedestrian_Detection_INT8_models_test_codes/deploy.prototxt", \
  ]

  caffemodel_list = [ \
    "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes/license_plate_recognition_quantizations.caffemodel", \
    "/wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes/car_logo_recognition_quantizations.caffemodel", \
    "/wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes/car_attributes_recognition_quantizations.caffemodel", \
    "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes/pedestrian_attributes_recognition_quantizations.caffemodel", \
    "/wrk/acceleration/models/deephi/reid_model_release_20190301/deploy.caffemodel", \
    "/wrk/acceleration/models/deephi/Car_Logo_Detection/deploy.caffemodel", \
    "/wrk/acceleration/models/deephi/Plate_Detection/deploy.caffemodel", \
#    "/wrk/acceleration/models/deephi/Pedestrian_Detection_INT8_models_test_codes/deploy.caffemodel", \
  ]

  quantcfg_list = [ \
    "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes/fix_info.txt", \
    "/wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes/fix_info.txt", \
    "/wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes/fix_info.txt", \
    "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes/fix_info.txt", \
    "/wrk/acceleration/models/deephi/reid_model_release_20190301/fix_info.txt", \
    "/wrk/acceleration/models/deephi/Car_Logo_Detection/fix_info.txt", \
    "/wrk/acceleration/models/deephi/Plate_Detection/fix_info.txt", \
#    "/wrk/acceleration/models/deephi/Pedestrian_Detection_INT8_models_test_codes/fix_info.txt", \
  ]

  return (prototxt_list,caffemodel_list,quantcfg_list)





def get_caffe_model_list_1():

  prototxt_list = [ \
    "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt" \
  ]

  caffemodel_list = [ \
    "models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel" \
  ]

  return (prototxt_list,caffemodel_list)



def get_caffe_model_list():
  (p,c,q) = get_caffe_model_list_all()
  return list(zip(p,c,q))

testdata = get_caffe_model_list()


import pytest

@pytest.mark.parametrize('prototxt,caffemodel,quantcfg', testdata)
def test_xfdnn_compiler_caffe_28(prototxt,caffemodel,quantcfg):
  print("Testing xfdnn_compiler_caffe...")

  dsp = 28
  mem = 4

  run_compiler(dsp,mem,prototxt,caffemodel,quantcfg)

@pytest.mark.parametrize('prototxt,caffemodel,quantcfg', testdata)
def test_xfdnn_compiler_caffe_56(prototxt,caffemodel,quantcfg):
  print("Testing xfdnn_compiler_caffe...")

  dsp = 56
  mem = 6

  run_compiler(dsp,mem,prototxt,caffemodel,quantcfg)

@pytest.mark.parametrize('prototxt,caffemodel,quantcfg', testdata)
def test_xfdnn_compiler_caffe_96(prototxt,caffemodel,quantcfg):
  print("Testing xfdnn_compiler_caffe...")

  dsp = 96
  mem = 9

  run_compiler(dsp,mem,prototxt,caffemodel,quantcfg)

if __name__ == "__main__":
  for (prototxt,caffemodel) in testdata:
    test_xfdnn_compiler_caffe_28(prototxt,caffemodel,quantcfg)
    test_xfdnn_compiler_caffe_56(prototxt,caffemodel,quantcfg)
    test_xfdnn_compiler_caffe_96(prototxt,caffemodel,quantcfg)


