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
import math
import sys
import timeit
import json
import xdnn, xdnn_io
import numpy as np
from collections import defaultdict

# example for multiple executors
def main(argv=None):
    args = xdnn_io.processCommandLine(argv)

    # processCommandLine()
    startTime = timeit.default_timer()
    ret = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", args['xlnxlib'])
    # ret = xdnn.createHandle(g_xclbin, "kernelSxdnn_0", g_xdnnLib)
    if ret != 0:
      sys.exit(1)
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to createHandle (%f ms):" % (elapsedTime * 1000)

    # we do not need other args keys except 'jsoncfg'
    args = args['jsoncfg']

    netCfgs   = defaultdict(dict)
    confNames = []
    for streamId, netCfg_args in enumerate(args):
      confName        = str(netCfg_args['name'])
      confNames      += [confName]
      netCfg_args['netcfg']         = './data/{}_{}.cmd'.format(netCfg_args['net'], netCfg_args['dsp'])
      netCfgs[confName]['streamId'] = streamId
      netCfgs[confName]['args']     = netCfg_args

    startTime = timeit.default_timer()
    # load weights for all networks simultaneously
    (weightsBlobs, 
        fcWeights,
         fcBiases)  = xdnn_io.loadWeights_samePE( args )

    # Does not matter which network to use. Both use the same input and PE.
    (fpgaInputs, batch_sz, shapes) = xdnn_io.prepareInput(netCfg_args, netCfg_args['PE'])

    for confName, netCfg in list(netCfgs.items()):
      netCfg['weightsBlobs'] = weightsBlobs[confName]
      netCfg['fcWeights']    = fcWeights[confName]
      netCfg['fcBiases']     = fcBiases[confName]
      netCfg['fpgaInputs']   = fpgaInputs
      netCfg['batch_sz']     = batch_sz
      netCfg['shapes']       = shapes
      netCfg['fpgaOutputs']  = xdnn_io.prepareOutput(netCfg['args']['fpgaoutsz'], batch_sz)
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to init (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    for confName, netCfg in list(netCfgs.items()):
      xdnn.exec_async(netCfg['args']['netcfg'],
                      netCfg['weightsBlobs'],
                      netCfg['fpgaInputs'],
                      netCfg['fpgaOutputs'],
                      netCfg['batch_sz'],
                      netCfg['args']['quantizecfg'],
                      netCfg['args']['scaleB'], 
                      netCfg['args']['PE'],
                      netCfg['streamId'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to Execonly (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    for confName, netCfg in list(netCfgs.items()):
      xdnn.get_result(netCfg['args']['PE'], netCfg['streamId'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to retrieve fpga outputs (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    for confName, netCfg in list(netCfgs.items()):
      fcOut = np.empty( (netCfg['batch_sz'] * netCfg['args']['outsz']), dtype=np.float32, order = 'C')
      xdnn.computeFC(netCfg['fcWeights'],
                     netCfg['fcBiases'],
                     netCfg['fpgaOutputs'],
                     netCfg['batch_sz'],
                     netCfg['args']['outsz'],
                     netCfg['args']['fpgaoutsz'],
                     fcOut)

      elapsedTime = timeit.default_timer() - startTime
      print "\nTime to FC (%f ms):" % (elapsedTime * 1000)

      startTime = timeit.default_timer()
      softmaxOut = xdnn.computeSoftmax(fcOut, netCfg['batch_sz'])
      elapsedTime = timeit.default_timer() - startTime
      print "\nTime to Softmax (%f ms):" % (elapsedTime * 1000)
    
      xdnn_io.printClassification(softmaxOut, netCfg['args']);

    print "\nSuccess!\n"

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
            --images dog.jpg \
            --labels synset_words.txt \
            --xlnxlib {5} \
            --jsoncfg data/multinet_singlePE.json".format(XCLBIN_PATH, DSP_WIDTH, 112/DSP_WIDTH, BITWIDTH, 2+DSP_WIDTH/14, LIBXDNN_PATH)

  argv = re.split(r'(?<!,)\s+', argv)
  '''

  main(argv)

