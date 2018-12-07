#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import sys
import xdnn, xdnn_io
import numpy as np

import timeit


def main():
  args= xdnn_io.processCommandLine()
  
  ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0")
  if ret != 0:
    sys.exit(1)
  fpgaRT = xdnn.XDNNFPGAOp(handles, args)
  fcWeight, fcBias = xdnn_io.loadFCWeightsBias(args)
  img_paths = xdnn_io.getFilePaths(args['images'])
  fpgaOutput = np.empty ((args['batch_sz'], args['fpgaoutsz'],), dtype=np.float32, order='C')
  fcOutput = np.empty((args['batch_sz'], args['outsz'],), dtype=np.float32, order='C')
  batch_array = np.empty(((args['batch_sz'],) + args['in_shape']), dtype=np.float32, order='C')
  labels = xdnn_io.get_labels(args['labels'])
  if args['golden']:
    goldenMap = xdnn_io.getGoldenMap(args['golden'])
    top5Count = 0
    top1Count = 0
    
  for i in xrange(0, len(img_paths), args['batch_sz']):
    pl = []
    for j, p in enumerate(img_paths[i:i + args['batch_sz']]):
      batch_array[j, ...], _ = xdnn_io.loadImageBlobFromFile(p, args['img_raw_scale'], args['img_mean'], args['img_input_scale'], args['in_shape'][2], args['in_shape'][1])
      pl.append(p)
      
    fpgaRT.execute(batch_array, fpgaOutput)
    xdnn.computeFC(fcWeight, fcBias, fpgaOutput, args['batch_sz'], args['outsz'], args['fpgaoutsz'], fcOutput)
    softmaxOut = xdnn.computeSoftmax(fcOutput)
    xdnn_io.printClassification(softmaxOut, pl, labels)
    if args['golden']:
      for j,p in enumerate(img_paths[i:i + args['batch_sz']]):
        top1Count += xdnn_io.isTopK(softmaxOut[j], goldenMap, p, labels, 1)
        top5Count += xdnn_io.isTopK(softmaxOut[j], goldenMap, p, labels, 5)
                
  xdnn.closeHandle()
  if args['golden']: 
    print ("\nAverage accuracy (n=%d) Top-1: %.1f%%, Top-5: %.1f%%\n") % (len(img_paths), float(top1Count)/float(len(img_paths))*100., 
    float(top5Count)/float(len(img_paths))*100.)        

if __name__ == '__main__':
    main()

