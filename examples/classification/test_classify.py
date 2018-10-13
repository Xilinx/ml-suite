#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import os.path
import math
import sys
import timeit
import xdnn, xdnn_io
import numpy as np
import types

def main():
  args = xdnn_io.processCommandLine()
  ret = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", args['xlnxlib'])
  if ret != 0:
    sys.exit(1)
  (weightsBlob, fcWeight, fcBias ) = xdnn_io.loadWeights( args )
  (fpgaInputs, batch_sz) = xdnn_io.prepareInput( args )
  fpgaOutput = xdnn_io.prepareOutput(args['fpgaoutsz'], batch_sz)
  for i in range(1):
    startTime = timeit.default_timer()
    xdnn.execute(args['netcfg'], 
      weightsBlob, fpgaInputs, fpgaOutput, 
      batch_sz, # num batches
      args['quantizecfg'], args['scaleB'], args['PE'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nAfter FPGA (%f ms)" % (elapsedTime*1000)

  startTime = timeit.default_timer()
  fcOut = xdnn.computeFC(fcWeight, fcBias, fpgaOutput, 
    batch_sz, args['outsz'], args['fpgaoutsz'], args['useblas'])
  elapsedTime = timeit.default_timer() - startTime
  print "\nAfter FC (%f ms)" % (elapsedTime*1000)
  #for i in range(10):
  #  print "%f" % fpgaOutput[i],

  startTime = timeit.default_timer()
  softmaxOut = xdnn.computeSoftmax(fcOut, batch_sz)
  elapsedTime = timeit.default_timer() - startTime
  print "\nAfter Softmax (%f ms)" % (elapsedTime*1000)
  
  #for i in range(10):
  #  print "%f" % fpgaOutput[i],

  xdnn_io.printClassification(softmaxOut, args);

  print "\nSuccess!\n"
  xdnn.closeHandle()
  
if __name__ == '__main__':
  main()

