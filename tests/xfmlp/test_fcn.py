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

def test_multiInstrv1(int_range, m, k, n, add_bias=False):
    print ("test_multiInstrv1: %d %d %d %d" % (int_range, m, k, n)) 
    A = np.random.randint(low=-int_range, high=int_range, size=(m, k), dtype=np.int16)
    B = np.random.randint(low=-int_range, high=int_range, size=(k, n), dtype=np.int16)
    C = np.zeros ((m, n), dtype=np.int16);
    D = np.random.randint(low=-int_range, high=int_range, size=(m, k), dtype=np.int16)
    E = np.zeros ((m, n), dtype=np.int16);
    b0 = np.zeros ((m, n), dtype=np.int32);        
    b1 = np.zeros ((m, n), dtype=np.int32);    
    if add_bias == True:
        b0 = np.random.randint(low=-int_range, high=int_range, size=(m, n), dtype=np.int32)
        b1 = np.random.randint(low=-int_range, high=int_range, size=(m, n), dtype=np.int32)
    xfmlp.sendMat(A)
    xfmlp.sendMat(B)
    xfmlp.sendMat(b0)
    xfmlp.sendMat(C)
    xfmlp.sendMat(D)    
    xfmlp.sendMat(E)
    xfmlp.sendMat(b1)         
    xfmlp.addFCNOp(A, B, C, b0, 1, 13, 307, 10)
    xfmlp.addFCNOp(D, C, E, b1, 1, 18, 307, 10)
    xfmlp.execute()
    xfmlp.clearInstrBuf()
    xfmlp.getMat(C)
    xfmlp.getMat(E)
    print("test C")
    test.multiply_and_cmp(C, A, B, b0, m, n, [1, 13],[307, 10])
    print("test E")
    test.multiply_and_cmp(E, D, C, b1, m, n, [1, 18],[307, 10])
      
if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test=FcnTest()
  args, xclbin_opts = xfmlp.processCommandLine()
  xfmlp.createFCNHandle( args, xclbin_opts)

  for j in range (1,3):
      for k in range(1,8):
          #for n in range(1000):
              for i in range (int(xclbin_opts["GEMX_numKernels"])):
                  for m,n in ( [0,0], [1,0]):
                      test.test_basic_randint( i, xclbin_opts, [j,k], [m,n], 2048)    
      
  # test.test_rand_basic (32764, 0, 5, [1,0]) # larger matrix size will lead to hw timeout error in regression test
  test_multiInstrv1(32764, 512, 512, 128, True) 
      
  #xfmlp.printStats() 
  
