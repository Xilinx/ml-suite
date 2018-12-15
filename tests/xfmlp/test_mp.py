#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import numpy as np
import xfmlp
import sys
import random
import argparse
import time
import test
from test import FcnTest
import multiprocessing as mp
import time

class Processor(object):
  def __init__(self, args, xclbin_opts):
    np.random.seed(123)  # for reproducibility
    print ("Processor init!")
    xfmlp.createFCNHandle( args, xclbin_opts )
    cur = mp.current_process()
    print ('Loaded xclbin!!!!!!!!!!', cur.name, int(cur._identity[0]-1))
    self._test=FcnTest()
    self._xclbin_opts = xclbin_opts

  def run(self, post_scale, relu_scale, max_dim):
    cur = mp.current_process()
    print ('running:', cur.name, int(cur._identity[0]-1))
    pid = int(cur._identity[0])-1
    self._test.test_basic_randint( pid, self._xclbin_opts, post_scale, relu_scale, max_dim)    

def init(args, xclbin_opts):
    global inst
    inst = Processor(args, xclbin_opts)

def run ( post_scale, relu_scale, max_dim):
    return inst.run(post_scale, relu_scale, max_dim)

if __name__ == '__main__':
  #np.random.seed(123)  # for reproducibility
  args, xclbin_opts = xfmlp.processCommandLine()
#  xfmlp.createFCNHandle( args, xclbin_opts)
  p = mp.Pool( initializer = init, initargs = [args,xclbin_opts],processes = int(xclbin_opts["GEMX_numKernels"]) ) 
  #p = mp.Pool( initializer = init, initargs = [args,xclbin_opts],processes = 1 )
  for j in range (5):
      for k in range(5):
          for m,n in ( [0,0], [1,0]):
              p.apply_async ( run, [[j,k], [m,n], 2048] )

  p.close()     
  p.join() 
