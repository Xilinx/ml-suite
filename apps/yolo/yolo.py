##################################################################################
# Copyright (c) 2017, Xilinx, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################################################

import os,sys

from xyolo import xyolo

# Bring in the COCO API for managing the coco dataset
#sys.path.insert(0,os.path.abspath("../../../cocoapi/PythonAPI"))
#from pycocotools.coco import COCO

# Bring in Xilinx Compiler, and Quantizer
# We directly compile the entire graph to minimize data movement between host, and card
sys.path.insert(0,os.path.abspath("../../"))
from xfdnn.tools.compile.frontends.frontend_caffe  import CaffeFrontend as xfdnnCompiler
from xfdnn.tools.quantize.frontends.frontend_caffe import CaffeFrontend as xfdnnQuantizer

# Select Configuration
from configs import *

weights = "../../models/yolov2/caffe/fp32/yolov2.caffemodel" 

# Define the compiler, and its parameters
compiler = xfdnnCompiler(
  verbose=False,
  network_file=config["network_file"], # Prototxt filename: input file
  generate_file=config["netcfg"],      # Script filename: output file
  strategy="all",                      # Strategy for memory allocation
  memory=5,                            # Available on chip ram within xclbin  
  dsp=56,                              # Rows in DSP systolic array within xclbin
  ddr=256,                             # Memory to allocate in FPGA DDR for activation spill
  weights=weights                      # Floating Point weights, compiler will convert to framework agnostic directory structure
  )

# Define the quantizer, and its parameters
quantizer = xfdnnQuantizer(
  deploy_model=config["network_file"],           # Prototxt filename: input file
  weights=weights,                               # Floating Point weights
  calibration_directory="calibration_directory", # Directory containing calbration images
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

imgDir = "calibration_directory"

images = sorted([os.path.join(imgDir,name) for name in os.listdir(imgDir)])

batch_sz = 4 # This determines how many images will be preprocessed and migrated to FPGA DDR at a time

nbatches = len(images) / batch_sz # Ignore the remainder for now (Don't operate on partial batch)

# Define the xyolo instance
with xyolo(batch_sz=batch_sz,in_shape=config["dims"],quantizecfg=config["quantizecfg"],xclbin=xclbin,verbose=True) as detector:
  for i in range(nbatches):
    # Invoke detector
    detector.detect(images[i*batch_sz:(i+1)*batch_sz],display=False,coco=False)
    print("Finished batch %d" % (i+1))

  detector.stop()
