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
    ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0")
    # ret = xdnn.createHandle(g_xclbin, "kernelSxdnn_0", g_xdnnLib)
    if ret != 0:
      sys.exit(1)
    labels = xdnn_io.get_labels(args['labels'])

    # TODO dict of tuples instead?
    fpgaRT          = {}
    fpgaOutputs     = {}
    fcWeights       = {}
    fcBiases        = {}
    netFiles        = {}
    confNames       = []

    args = args['jsoncfg']      # we do not use other args' keys
    for netconf_args in args:
      confName   = str(netconf_args['name'])
      confNames += [confName]
      netconf_args['netcfg'] = './data/{}_{}.json'.format(netconf_args['net'], netconf_args['dsp'])
      fpgaRT[confName] = xdnn.XDNNFPGAOp(handles, netconf_args)
      (fcWeights[confName],
        fcBiases[confName]) = xdnn_io.loadFCWeightsBias(netconf_args)
      fpgaOutputs[confName]             = np.empty ((netconf_args['batch_sz'], int(netconf_args['fpgaoutsz']),), dtype=np.float32, order='C')
      netFiles[confName]                = str(netconf_args['netcfg'])
    
    for streamId, netconf_args in enumerate(args):
      batch_array = np.empty((( netconf_args['batch_sz'],) + netconf_args['in_shape']), dtype=np.float32, order='C')
      pl = []
      img_paths = xdnn_io.getFilePaths(netconf_args['images'])
      for j, p in enumerate(img_paths[:netconf_args['batch_sz']]):
        batch_array[j, ...], _ = xdnn_io.loadImageBlobFromFile(p, netconf_args['img_raw_scale'], 
                                                                  netconf_args['img_mean'],
                                                                  netconf_args['img_input_scale'], 
                                                                  netconf_args['in_shape'][1], 
                                                                  netconf_args['in_shape'][2])
        pl.append(p)
          
      confName = str(netconf_args['name'])
      fpgaRT[confName].exec_async(batch_array, fpgaOutputs[confName], streamId)
    
    for streamId, confName in enumerate(confNames):
      fpgaRT[confName].get_result (streamId)
    
    for netconf_args in args:
      confName = str(netconf_args['name'])
      fcOut = np.empty( (netconf_args['batch_sz'], netconf_args['outsz']), dtype=np.float32, order = 'C')
      xdnn.computeFC (fcWeights[confName], fcBiases[confName], fpgaOutputs[confName],
                              netconf_args['batch_sz'], netconf_args['outsz'], netconf_args['fpgaoutsz'], fcOut)
    
      softmaxOut = xdnn.computeSoftmax(fcOut)
      xdnn_io.printClassification(softmaxOut, netconf_args['images'], labels);

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

