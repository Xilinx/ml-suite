#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import sys
import timeit
import xdnn, xdnn_io
import numpy as np
from operator import mul


def benchmark():

  mode = "Non-Blocking"
  #mode = "Blocking"

  # Extract Arguments from json
  args = xdnn_io.processCommandLine()["jsoncfg"][0]

  if "platform" in args:
    args["xclbin"] = "../../overlaybins/" + str(args["platform"]) + "/" + args["xclbin"]

  # Establish Communication w/ FPGA
  if xdnn.createHandle(args['xclbin'],libFile=args['xlnxlib']):
    sys.exit(1)
  
  # Transfer weights to device memory
  if "usexdnnv3" in args and args["usexdnnv3"] == "1":
    weightsBlob = xdnn_io.loadWeightsBiasQuantv3(args)
  else:  
    weightsBlob = xdnn_io.loadWeightsBiasQuant(args)

  # Create random input data
  fpgaInputs = []
  fpgaInputs.append(np.float32(np.random.standard_normal((args["batchsz"],reduce(mul,args["in_shape"],1)))))
  fpgaInputs[0] = xdnn.quantizeInputs(args["firstfpgalayer"], args["quantizecfg"], args["scaleB"], fpgaInputs[0])
  fpgaInputs[0] = xdnn.prepareInputsForFpga(fpgaInputs[0], args["quantizecfg"], args["scaleB"], -1, args["firstfpgalayer"],0)
  fpgaInputs.append(np.float32(np.random.standard_normal((args["batchsz"],reduce(mul,args["in_shape"],1)))))
  fpgaInputs[1] = xdnn.quantizeInputs(args["firstfpgalayer"], args["quantizecfg"], args["scaleB"], fpgaInputs[1])
  fpgaInputs[1] = xdnn.prepareInputsForFpga(fpgaInputs[1], args["quantizecfg"], args["scaleB"], -1, args["firstfpgalayer"],1)

  # Create buffers in host memory for result
  fpgaOutputs = []
  fpgaOutputs.append(xdnn_io.prepareOutput(args['fpgaoutsz'], args["batchsz"]))
  fpgaOutputs.append(xdnn_io.prepareOutput(args['fpgaoutsz'], args["batchsz"]))
 
  # Load network schedule to accelerator
  xdnn.initScript(args['netcfg'], weightsBlob, args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'],0)
  xdnn.initScript(args['netcfg'], weightsBlob, args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'],1)
 
  # Run forward propagation N times
  print("Running inference...\n")
  cumulative_time = -1*timeit.default_timer()
  

  if mode == "Non-Blocking":

    xdnn.exec_async(args['netcfg'], weightsBlob, fpgaInputs[0], fpgaOutputs[0], args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'],0)
    xdnn.exec_async(args['netcfg'], weightsBlob, fpgaInputs[1], fpgaOutputs[1], args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'],1)

    for i in range(args["iterations"]/2-1):
      xdnn.get_result(-1, 0) # get 0
      xdnn.exec_async(args['netcfg'], weightsBlob, fpgaInputs[0], fpgaOutputs[0], args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'],0) # push 0
      xdnn.get_result(-1, 1) # get 1
      xdnn.exec_async(args['netcfg'], weightsBlob, fpgaInputs[1], fpgaOutputs[1], args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'],1) # push 1
    
    xdnn.get_result(-1, 0) # get 0
    xdnn.get_result(-1, 1) # get 1
    
  else:
    for i in range(args["iterations"]):
      xdnn.execute(args['netcfg'], weightsBlob, fpgaInputs[0], fpgaOutputs[0], args["batchsz"], args['quantizecfg'], args['scaleB'], args['PE'])

  cumulative_time += timeit.default_timer()

  # Summarize
  print("===========================================")
  print("Performance Summary\n")
  print("  Network: %s" % (args["name"]))
  print("  Precision: %d" % (args["precision"]))
  print("  Images: %d" % (args["iterations"]*args["batchsz"])) 
  print("  Batch Size: %d" % (args["batchsz"])) 
  print("  Total Batches: %d" % (args["iterations"])) 
  print("  Total Time: %.2f ms" % (1000*cumulative_time))
  print("  SIL: %.2f ms" % (1000*cumulative_time/args["iterations"])) # Time per batch # Single Image Latency
  print("  FPS: %.2f" % (args["iterations"]*args["batchsz"]/cumulative_time))
  print("  GOPS: %.2f" % (args["ops"]*args["iterations"]*args["batchsz"]/cumulative_time/1000000000))
  print("===========================================\n")

  # Release FPGA
  xdnn.closeHandle()
  
if __name__ == '__main__':
  benchmark()

