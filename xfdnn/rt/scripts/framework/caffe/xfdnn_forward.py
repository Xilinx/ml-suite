#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import numpy as np
import caffe
import argparse
#import copy

def forward(prototxt,caffemodel,numBatches): 
  net = caffe.Net(prototxt,caffemodel,caffe.TEST)
  inputKey = net.blobs.keys()[0]
  ptxtShape = net.blobs[inputKey].data.shape
  print "Running with shape of: ",ptxtShape
  accum = {}
  for i in xrange(1,numBatches+1): # 1000 iterations w/ batchSize 50 will yield 50,000 images 
    out = net.forward()
    for k in out:
      if out[k].size != 1:
        continue
      if k not in accum:
        accum[k] = 0.0 
      accum[k] += out[k]
      print k, " -- This Batch: ",out[k]," Average: ",accum[k]/i," Batch#: ",i

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXDNN')
  parser.add_argument('--prototxt', default="", help='User must provide the prototxt')
  parser.add_argument('--caffemodel', default="", help='User must provide the caffemodel')
  parser.add_argument('--numBatches', type=int, default=1000, help='User must provide the caffemodel')
  args = vars(parser.parse_args())
  forward(args["prototxt"],args["caffemodel"],args["numBatches"])
