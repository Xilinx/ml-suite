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

#!/usr/bin/python

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

