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

# Example for asynchronous multi-net classification using xdnn. Derived from test_hclassify.py
# 2017-11-09 22:53:07 parik

import argparse
import os.path
import math
import sys
import timeit
import json
import xdnn, xdnn_io
import numpy as np

# example for multiple executors
def main():
    args = xdnn_io.processCommandLine()
    
    # processCommandLine()
    startTime = timeit.default_timer()
    ret = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", args['xlnxlib'])
    # ret = xdnn.createHandle(g_xclbin, "kernelSxdnn_0", g_xdnnLib)
    if ret != 0:
      sys.exit(1)
    elapsedTime = timeit.default_timer() - startTime
    print "\nAfter createHandle (%f ms):" % (elapsedTime * 1000)
    startTime = timeit.default_timer()
    
    # TODO dict of tuples instead?
    fpgaInputs = {}
    fpgaOutputs = {}
    weightsBlobs = {}
    fcWeights = {}
    fcBiases = {}
    batch_sizes = {}
    fpgaOutputSizes = {}
    PEs = {}
    netFiles = {}
    confNames = []
    
    for netconf_args in args['jsoncfg']:
      confName = str(netconf_args['name'])
      confNames.append (confName)
      # make a tuple instead
      PE = [int(x) for x in netconf_args['PE'].split()]
      # if cuMask in cuMaskList:
      #  raise Exception('cuMasks are non-disjoint')
      datadir = str(netconf_args['datadir'])
      fpgaoutsz = int(netconf_args['fpgaoutsz'])
      netfile = str(netconf_args['netcfg'])
    
      PEs [confName] = PE
      (weightsBlobs[confName], fcWeights [confName], fcBiases [confName] ) = xdnn_io.loadWeights( netconf_args )
      fpgaOutputSizes[confName] = fpgaoutsz
      (fpgaInputs[confName], batch_sz) = xdnn_io.prepareInput(netconf_args, PE)
      batch_sizes[confName] = batch_sz
      fpgaOutputs [confName] = xdnn_io.prepareOutput(int(netconf_args['fpgaoutsz']) , batch_sz)
      netFiles [confName] = netfile
    
    elapsedTime = timeit.default_timer() - startTime
    print "\nAfter init (%f ms):" % (elapsedTime * 1000)
    startTime = timeit.default_timer()

    for netconf_args in args['jsoncfg']:
      confName = str(netconf_args['name'])
      xdnn.exec_async (netFiles [confName], weightsBlobs [confName], fpgaInputs [confName],
        fpgaOutputs [confName], int(batch_sizes[confName]), netconf_args['quantizecfg'], netconf_args['scaleB'], PEs [confName])
    
    elapsedTime = timeit.default_timer() - startTime
    print "\nAfter Execonly (%f ms):" % (elapsedTime * 1000)
    startTime = timeit.default_timer()
    
    for confName in confNames:
      xdnn.get_result (PEs [confName])
    
    elapsedTime = timeit.default_timer() - startTime
    print "\nAfter wait (%f ms):" % (elapsedTime * 1000)
    startTime = timeit.default_timer()
    
    for netconf_args in args['jsoncfg']:
      confName = str(netconf_args['name'])
      fcOut = xdnn.computeFC (fcWeights[confName], fcBiases[confName], fpgaOutputs[confName],
                              batch_sizes[confName], netconf_args['outsz'], netconf_args['fpgaoutsz'], netconf_args['useblas'])
    
      elapsedTime = timeit.default_timer() - startTime
      print "\nAfter FC (%f ms):" % (elapsedTime * 1000)
      startTime = timeit.default_timer()
    
      softmaxOut = xdnn.computeSoftmax(fcOut, batch_sizes[confName])
    
      elapsedTime = timeit.default_timer() - startTime
      print "\nAfter Softmax (%f ms):" % (elapsedTime * 1000)
    
      xdnn_io.printClassification(softmaxOut, netconf_args);

    print "\nSuccess!\n"

    xdnn.closeHandle()

if __name__ == '__main__':
  main()

