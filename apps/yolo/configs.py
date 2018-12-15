#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from __future__ import print_function

import os

def select_config(selection,platform="alveo-u200"):
  # User can choose to run YOLO at different input sizes
  # - 608x608
  # - 224x224
  # User can choose to run YOLO at different quantization precisions
  # - 16b
  # - 8b

  mlsuiteRoot = os.getenv("MLSUITE_ROOT", "../..")

  configs = {
     'rectangular_608_16b': 
               {'dims': [3, 456, 608], 
                'bitwidths': [16, 16, 16], 
                'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_rectangular_608.prototxt', 
                'netcfg': 'work/yolo.cmds', 
                'quantizecfg': 'work/yolo_deploy_rectangular_608_16b.json',
                'datadir': 'work/yolov2.caffemodel_data',
                'firstfpgalayer': 'conv0',
                'classes': 80,
                'memory' : 5,
                'dsp' : 56,
                'ddr' : 256,
                'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'},
     '608_16b': 
               {'dims': [3, 608, 608], 
                'bitwidths': [16, 16, 16], 
                'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_608.prototxt', 
                'netcfg': 'work/yolo.cmds', 
                'quantizecfg': 'work/yolo_deploy_608_16b.json',
                'datadir': 'work/yolov2.caffemodel_data',
                'firstfpgalayer': 'conv0',
                'classes': 80,
                'memory' : 5,
                'dsp' : 56,
                'ddr' : 256,
                'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'},
     '608_8b': {'dims': [3, 608, 608], 
                'bitwidths': [8, 8, 8], 
                'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_608.prototxt', 
                'netcfg': 'work/yolo.cmds', 
                'quantizecfg': 'work/yolo_deploy_608_8b.json',
                'datadir': 'work/yolov2.caffemodel_data',
                'firstfpgalayer': 'conv0',
                'classes': 80,
                'memory' : 5,
                'dsp' : 56,
                'ddr' : 256,
                'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'},
     '416_16b': 
               {'dims': [3, 416, 416], 
                'bitwidths': [16, 16, 16], 
                'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_416.prototxt', 
                'netcfg': 'work/yolo.cmds', 
                'quantizecfg': 'work/yolo_deploy_416_16b.json',
                'datadir': 'work/yolov2.caffemodel_data',
                'firstfpgalayer': 'conv0',
                'classes': 80,
                'memory' : 5,
                'dsp' : 56,
                'ddr' : 256,
                'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'},
     '416_8b': 
               {'dims': [3, 416, 416], 
                'bitwidths': [8, 8, 8], 
                'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_416.prototxt', 
                'netcfg': 'work/yolo.cmds', 
                'quantizecfg': 'work/yolo_deploy_416_8b.json',
                'datadir': 'work/yolov2.caffemodel_data',
                'firstfpgalayer': 'conv0',
                'classes': 80,
                'memory' : 5,
                'dsp' : 56,
                'ddr' : 256,
                'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'},
     '224_16b': {'dims': [3, 224, 224], 
                'bitwidths': [16, 16, 16], 
                'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_224.prototxt', 
                'netcfg': 'work/yolo.cmds', 
                'quantizecfg': 'work/yolo_deploy_224_16b.json',
                'datadir': 'work/yolov2.caffemodel_data',
                'firstfpgalayer': 'conv0',
                'classes': 80,
                'memory' : 5,
                'dsp' : 56,
                'ddr' : None,
                'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'},
      '224_8b': {'dims': [3, 224, 224], 
                 'bitwidths': [8, 8, 8], 
                 'network_file': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolo_deploy_224.prototxt', 
                 'netcfg': 'work/yolo.cmds', 
                 'quantizecfg': 'work/yolo_deploy_224_8b.json',
                 'datadir': 'work/yolov2.caffemodel_data',
                 'firstfpgalayer': 'conv0',
                 'classes': 80,
                 'memory' : 5,
                 'dsp' : 56,
                 'ddr' : None,
                 'weights': mlsuiteRoot+'/models/caffe/yolov2/fp32/yolov2.caffemodel'}}
  
  # Choose a config here
  if selection in configs:
    config = configs[selection]
    aws = False
    eb = False
    if config['bitwidths'][0] == 8:
      eb = True
    if os.path.exists('/sys/hypervisor/uuid'):
      with open('/sys/hypervisor/uuid') as (file):
        contents = file.read()
        if 'ec2' in contents:
          print('Running on Amazon AWS EC2')
          aws = True
          if eb:
            config["xclbin"] = mlsuiteRoot+'/overlaybins/aws/overlay_2.xclbin'
          else:
            config["xclbin"] = mlsuiteRoot+'/overlaybins/aws/overlay_3.xclbin'
    if not aws:
      if eb:
        config["xclbin"] = mlsuiteRoot+'/overlaybins/' + platform + '/overlay_2.xclbin'
      else:
        config["xclbin"] = mlsuiteRoot+'/overlaybins/' + platform + '/overlay_3.xclbin'
    return config
  else:
    print("Error: You chose an invalid configuration")
    return None
