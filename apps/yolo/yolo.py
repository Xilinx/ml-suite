#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import os,sys

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
print(sys.path)

from xyolo import xyolo

# Bring in the COCO API for managing the coco dataset
#sys.path.insert(0,os.path.abspath("../../../cocoapi/PythonAPI"))
#from pycocotools.coco import COCO

# Bring in Xilinx Compiler, and Quantizer
# We directly compile the entire graph to minimize data movement between host, and card
from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizer

# Select Configuration
from configs import select_config

config = select_config("608_16b",sys.argv[1])

mlsuiteRoot = os.getenv("MLSUITE_ROOT", "../..")

# Define the compiler, and its parameters
compiler = xfdnnCompiler(
  verbose=False,
  networkfile=config["network_file"], # Prototxt filename: input file
  generatefile=config["netcfg"],      # Script filename: output file
  strategy="all",                      # Strategy for memory allocation
  memory=config["memory"],                            # Available on chip ram within xclbin  
  dsp=config["dsp"],                              # Rows in DSP systolic array within xclbin
  ddr=config["ddr"],                             # Memory to allocate in FPGA DDR for activation spill
  weights=config["weights"]                      # Floating Point weights, compiler will convert to framework agnostic directory structure
  )

# Define the quantizer, and its parameters
quantizer = xfdnnQuantizer(
  deploy_model=config["network_file"],           # Prototxt filename: input file
  weights=config["weights"],                               # Floating Point weights
  output_json=config["quantizecfg"],
  calibration_directory=mlsuiteRoot+"/xfdnn/tools/quantize/calibration_directory", # Directory containing calbration images
  calibration_size=8,                            # Number of calibration images to use
  calibration_seed=None,                         # Seed for randomly choosing calibration images
  calibration_indices=None,                      # User can control which images to use for calibration [DEPRECATED]
  bitwidths=config["bitwidths"],                 # Fixed Point precision: 8b or 16b
  dims=config["dims"],                           # Image dimensions [Nc,Nw,Nh]
  transpose=[2,0,1],                             # Transpose argument to caffe transformer
  channel_swap=[2,1,0],                          # Channel swap argument to caffe transfomer
  raw_scale=1,                                   # Raw scale argument to caffe transformer
  mean_value=[0,0,0],                            # Image mean per channel to caffe transformer
  input_scale=1                                  # Input scale argument to caffe transformer
  )

# Invoke compiler
compiler.compile()

# Invoke quantizer
quantizer.quantize()

# For COCO Validation Test
# Depends on downloading the COCO Images
#imgDir = "../../../cocoapi/images/val2014"

imgDir = mlsuiteRoot+"/xfdnn/tools/quantize/calibration_directory"

images = sorted([os.path.join(imgDir,name) for name in os.listdir(imgDir)])

batch_sz = 4 # This determines how many images will be preprocessed and migrated to FPGA DDR at a time

nbatches = len(images) // batch_sz # Ignore the remainder for now (Don't operate on partial batch)

# Define the xyolo instance
with xyolo(batch_sz=batch_sz,in_shape=tuple(config["dims"]),quantizecfg=config["quantizecfg"], xlnxlib=mlsuiteRoot+"/xfdnn/rt/xdnn_cpp/lib/libxfdnn.so", xclbin=config["xclbin"],netcfg=config["netcfg"], datadir=config["datadir"],firstfpgalayer=config["firstfpgalayer"],classes=config["classes"],verbose=True) as detector:
  for i in range(nbatches):
    # Invoke detector
    detector.detect(images[i*batch_sz:(i+1)*batch_sz],display=False,coco=False)
    print("Finished batch %d" % (i+1))

  detector.stop()
