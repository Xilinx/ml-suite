#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import argparse
import os.path
import math
import sys
import timeit
import json
import xdnn, xdnn_io
import numpy as np
from ctypes import *

# example for multiple executors
def main(argv):
    args = xdnn_io.processCommandLine(argv)

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
    fpgaInputs      = {}
    fpgaOutputs     = {}
    #weightsBlobs    = {}
    #fcWeights       = {}
    #fcBiases        = {}
    batch_sizes     = {}
    PEs             = {}
    netFiles        = {}
    confNames       = []
    handles = {}

    args = args['jsoncfg']      # we do not use other args' keys
    (weightsBlobs,
        fcWeights,
         fcBiases) = xdnn_io.loadWeights_samePE(args) #xdnn_io.loadWeights( netconf_args )    
    for netconf_args in args:
      confName   = str(netconf_args['name'])
      confNames += [confName]
      PE         = [int(x) for x in str(netconf_args['PE']).split()]

      netconf_args['netcfg'] = './data/{}_{}.cmd'.format(netconf_args['net'], netconf_args['dsp'])

      (fpgaInputs[confName],
                   batch_sz, __)        = xdnn_io.prepareInput(netconf_args, PE)
      fpgaOutputs[confName]             = xdnn_io.prepareOutput(int(netconf_args['fpgaoutsz']) , batch_sz)
      batch_sizes[confName]             = batch_sz
      netFiles[confName]                = str(netconf_args['netcfg'])
      PEs[confName]                     = PE
    
    
      numHandles = len(xdnn._xdnnManager._handles)
      handlePtrs = (c_void_p*numHandles)()
      for i,h in enumerate(xdnn._xdnnManager._handles):
        handlePtrs[i] = h

    for netconf_args in args:
        confName = str(netconf_args['name'])
        print ( "PE: ", int(PEs[confName][0]) )
        handles[confName] = xdnn._xdnnManager._lib.XDNNMakeScriptExecutor(
            handlePtrs, numHandles, weightsBlobs[confName], netFiles[confName], netconf_args['quantizecfg'], 
            netconf_args['scaleB'], int(batch_sizes[confName]), 1, int(PEs[confName][0]))
          
    for netconf_args in args:
        confName = str(netconf_args['name'])
        outputPtr = fpgaOutputs[confName].ctypes.data_as(c_void_p)
        
        result = xdnn._xdnnManager._lib.XDNNExecute(handles[confName], 
                                       fpgaInputs[confName], outputPtr, int(batch_sizes[confName]), 0, True)      
      #xdnn.execute(netFiles[confName], weightsBlobs[confName], fpgaInputs[confName],
      #  fpgaOutputs[confName], int(batch_sizes[confName]), netconf_args['quantizecfg'], netconf_args['scaleB'], PEs[confName])
    
    for netconf_args in args:
      confName = str(netconf_args['name'])
      fcOut = np.empty( (batch_sizes[confName]* netconf_args['outsz']), dtype=np.float32, order = 'C')
      xdnn.computeFC (fcWeights[confName], fcBiases[confName], fpgaOutputs[confName],
                              batch_sizes[confName], netconf_args['outsz'], netconf_args['fpgaoutsz'], fcOut)
    
      softmaxOut = xdnn.computeSoftmax(fcOut, batch_sizes[confName])
      xdnn_io.printClassification(softmaxOut, netconf_args);

    xdnn.closeHandle()

if __name__ == '__main__':
  argv = None

  '''
  import os
  import re

  XCLBIN_PATH   = os.environ['XCLBIN_PATH']
  LIBXDNN_PATH  = os.environ['LIBXDNN_PATH']
  DSP_WIDTH     = 56
  BITWIDTH      = 8
  MLSUITE_ROOT  = os.environ['MLSUITE_ROOT']

  argv =   "--xclbin {0}/xdnn_v2_32x{1}_{2}pe_{3}b_{4}mb_bank21.xclbin \
            --labels synset_words.txt \
            --xlnxlib {5} \
            --jsoncfg data/multinet.json".format(XCLBIN_PATH, DSP_WIDTH, 112/DSP_WIDTH, BITWIDTH, 2+DSP_WIDTH/14, LIBXDNN_PATH, MLSUITE_ROOT)

  argv = re.split(r'(?<!,)\s+', argv)
  '''

  main(argv)

