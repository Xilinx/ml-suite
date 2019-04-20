#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import os,sys

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
print(sys.path)
from xfdnn.rt import xdnn_io
from xyolo import xyolo


# Bring in the COCO API for managing the coco dataset
#sys.path.insert(0,os.path.abspath("../../../cocoapi/PythonAPI"))
#from pycocotools.coco import COCO

# Bring in Xilinx Compiler, and Quantizer
# We directly compile the entire graph to minimize data movement between host, and card
from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizer

# Select Configuration
from configs import select_config

def main():
  #config= xdnn_io.processCommandLine()
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--in_shape', default=[3,224,224], nargs=3, type=int, help='input dimensions')
  parser.add_argument("--yolo_model",  type=str, default='xilinx_yolo_v2')
  parser.add_argument('--caffe_inference', help='.caffe prototxt for layers on caffe', type=str, metavar="FILE")
 
  args = parser.parse_args()
  config = xdnn_io.make_dict_args(args)
  mlsuiteRoot = os.getenv("MLSUITE_ROOT", "../..")
  
  # yolo_v2 and  coco dataset specific
  config['firstfpgalayer'] = 'conv0'
  config['ddr'] = 256
  #config['classes'] = 80
  
  IS_XDNN_V3 = False
  bit_width = int(config['overlaycfg']['XDNN_BITWIDTH'])
  string_to_config = str(config['in_shape'][1]) + "_" + str(bit_width) + "b"
  bytesperpixels_val=2
  if(int(config['overlaycfg']['XDNN_VERSION_MAJOR']) == 3):
      string_to_config = string_to_config + "_v3"
      if(bit_width != 16):
          bytesperpixels_val = 1
      IS_XDNN_V3 = True
      
  print(string_to_config) 
  print(config) 
  dummy_config = select_config(string_to_config,sys.argv[1])
  config['dims'] = dummy_config['dims']
  config['bitwidths'] = dummy_config['bitwidths']
  config['memory'] = dummy_config['memory']
  
  
  run_quantizer = False
  if(config['quantizecfg'].split(".")[-1] != "json"):
      config['quantizecfg'] = dummy_config['quantizecfg']
      run_quantizer = True
      

  run_complier = False
  hand_coded_json = None
  # if dummy file as given as netcfg file
  if(config['netcfg'].split(".")[-1] != "json"):
      config['netcfg'] = dummy_config['netcfg']
      config['weights'] = dummy_config['datadir']
      run_complier = True
  elif(os.path.isdir(config['weights']) == False):
      # for handcode json file, compiler is still needed to run generate wieght files
      hand_coded_json = config['netcfg']
      config['netcfg'] = dummy_config['netcfg']
      config['weights'] = dummy_config['datadir']
      run_complier = True
       
  
  print("after dummy config")
  print(config) 
 
                
  
  # Define the compiler, and its parameters
  compiler = xfdnnCompiler(
            verbose=False,
            networkfile=config["net_def"], # Prototxt filename: input file
            generatefile=config["netcfg"],      # Script filename: output file
            strategy="all",                      # Strategy for memory allocation
            memory=config["memory"],                            # Available on chip ram within xclbin  
            dsp=config["dsp"],                              # Rows in DSP systolic array within xclbin
            ddr=config["ddr"],                             # Memory to allocate in FPGA DDR for activation spill
            weights=config["net_weights"],                      # Floating Point weights, compiler will convert to framework agnostic directory structure
            bytesperpixels=bytesperpixels_val,
            #pipelineconvmaxpool=IS_XDNN_V3,
            cpulayermustgo=True,
            parallelread="all"
            #godreplication='/home/arunkuma/elliot.godreplication.csv'
            #noreplication=True
            )
    
    
  # Define the quantizer, and its parameters
  quantizer = xfdnnQuantizer(
            xdnn_version=int(config['overlaycfg']['XDNN_VERSION_MAJOR']),
            deploy_model=config["net_def"],           # Prototxt filename: input file
            weights=config["net_weights"],                               # Floating Point weights
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
  if run_complier:
      compiler.compile()
    
  # Invoke quantizer
  if run_quantizer:
      quantizer.quantize()

  if hand_coded_json is not None:
      config["netcfg"] = hand_coded_json
      
  # strip .json if it exisits at end
  if(config["netcfg"].endswith('.json')):
      config["netcfg"] = config["netcfg"][:-5]
    
  #imgDir = mlsuiteRoot+"/xfdnn/tools/quantize/calibration_directory"
  #config["netcfg"] = 'work/yolo416.cmds'
  #config["netcfg"] = 'work/yolo608.cmds'
  #config["netcfg"] = 'work/tinyyolo.1.96.9.viw' # for 224
    
  images = xdnn_io.getFilePaths(config['images'])  

  batch_sz = config['batch_sz'] # This determines how many images will be preprocessed and migrated to FPGA DDR at a time

  # pad image list to batch_sz so we don't skip partial batches
  if len(images):
    idx = 0
    while len(images) % batch_sz:
      images.append(images[idx])
      idx += 1
    
  nbatches = len(images) // batch_sz # Ignore the remainder for now (Don't operate on partial batch)
    
  # Define the xyolo instance
  with xyolo(batch_sz=batch_sz,in_shape=tuple(config["dims"]),quantizecfg=config["quantizecfg"],
             xlnxlib="/wrk/acceleration/users/arun/MLsuite_yolo/xfdnn/rt/libs/libxfdnn.so.2.20182.v3",
             xclbin=config["xclbin"],netcfg=config["netcfg"]+".json", weights=config["weights"],
             firstfpgalayer=config["firstfpgalayer"],classes=config["outsz"],verbose=True,
             yolo_model=config["yolo_model"],
             caffe_prototxt=config["caffe_inference"],
             caffe_model=config["net_weights"]) as detector:
      
      for i in range(nbatches):
          # Invoke detector
          detector.detect(images[i*batch_sz:(i+1)*batch_sz],display=True,coco=False)
          print("Finished batch %d" % (i+1))
            
      detector.stop()

if __name__ == '__main__':
    main()
