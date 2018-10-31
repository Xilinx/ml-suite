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
      PE = [int(x) for x in str(netconf_args['PE']).split()]
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

    for streamId, netconf_args in enumerate(args['jsoncfg']):
      confName = str(netconf_args['name'])
      xdnn.exec_async (netFiles [confName], weightsBlobs [confName], fpgaInputs [confName],
        fpgaOutputs [confName], int(batch_sizes[confName]), netconf_args['quantizecfg'], netconf_args['scaleB'], PEs [confName], streamId)
    
    elapsedTime = timeit.default_timer() - startTime
    print "\nAfter Execonly (%f ms):" % (elapsedTime * 1000)
    startTime = timeit.default_timer()
    
    for streamId, confName in enumerate(confNames):
      xdnn.get_result (PEs [confName], streamId)
    
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
  argv = None

  '''
  import os
  import re

  XCLBIN_PATH = os.environ['XCLBIN_PATH']
  LIBXDNN_PATH = os.environ['LIBXDNN_PATH']
  DSP_WIDTH = 56
  BITWIDTH  = 16
  MLSUITE_ROOT = os.environ['MLSUITE_ROOT']

  argv =   "--xclbin {0}/xdnn_v2_32x{1}_{2}pe_{3}b_{4}mb_bank21.xclbin \
            --labels synset_words.txt \
            --xlnxlib {5} \
            --jsoncfg data/multinet.json".format(XCLBIN_PATH, DSP_WIDTH, 112/DSP_WIDTH, BITWIDTH, 2+DSP_WIDTH/14, LIBXDNN_PATH, MLSUITE_ROOT)

  argv = re.split(r'(?<!,)\s+', argv)
  '''

  main(argv)