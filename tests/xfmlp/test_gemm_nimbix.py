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
from test import GemmTest

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
    xfmlp.addGEMMOp(A, B, C, b0, 1, 0)
    xfmlp.addGEMMOp(D, C, E, b1, 1, 0)
    xfmlp.execute()
    xfmlp.clearInstrBuf()
    xfmlp.getMat(C)
    xfmlp.getMat(E)
    print("test C")
    test.multiply_and_cmp(C, A, B, b0, m, n, [1, 0])
    print("test E")
    test.multiply_and_cmp(E, D, C, b1, m, n, [1, 0])
    
def test_perf_gemm(m, k, n, A_range=32764, B_range=32764, bias_range=32764, post_scale=[1,0]):
    mat_A = np.random.randint(low=-A_range, high=A_range, size=(m, k), dtype=np.int16)
    mat_B = np.random.randint(low=-B_range, high=B_range, size=(k, n), dtype=np.int16)  
    bias = []
    if bias_range != 0:
        bias = np.random.randint(low=-bias_range, high=bias_range, size=(m, n), dtype=np.int32)
    else:
        bias = np.zeros ((m, n), dtype=np.int32, order='C');   
    C_fpga = np.zeros( (m, n), dtype=np.int16)
    timePointKernel = []
    timePointKernel.append(time.time()) # current time    
    xfmlp.sendMat(mat_A)
    xfmlp.sendMat(mat_B)
    xfmlp.sendMat(C_fpga)    
    xfmlp.sendMat(bias)
    xfmlp.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1])
    timePointKernel.append(time.time()) # send to FPGA
    xfmlp.execute()
    xfmlp.clearInstrBuf()
    timePointKernel.append(time.time()) # call kernel
    xfmlp.getMat(C_fpga)  
    timePointKernel.append(time.time()) # copy from FPGA
    total_operations = 2 * m * n * k + m * n * 3
    total_parallel_operations = 2 * m * n * k
    freq = xfmlp.getFreq()
    test.test_perf(timePointKernel,total_operations,total_parallel_operations,freq,m,k,n)
    test.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
      
def test_perf_multi_gemm(ins_count, m_size, k_size, n_size, A_range, B_range, post_scale):
    total_operations = 0
    total_parallel_operations = 0
    mat_A=[]
    mat_C=[]
    mat_bias=[]
    for i in range(ins_count):
      total_operations += 2 * m_size[i] * n_size[i] * k_size[i] + m_size[i] * n_size[i] * 3
      total_parallel_operations += 2 * m_size[i] * n_size[i] * k_size[i]
      mat_A.append(np.random.randint(low=-A_range, high=A_range, size=(m_size[i], k_size[i]), dtype=np.int16))
      mat_bias.append(np.zeros ((m_size[i], n_size[i]), dtype=np.int32))
      mat_C.append(np.zeros((m_size[i], n_size[i]), dtype=np.int16, order='C'))
    mat_B0 = np.random.randint(low=-B_range, high=B_range, size=(k_size[0], n_size[0]), dtype=np.int16) 
    timePointKernel = []
    timePointKernel.append(time.time()) # current time 
    for i in range(ins_count):
      xfmlp.sendMat(mat_A[i])
      xfmlp.sendMat(mat_C[i])
      xfmlp.sendMat(mat_bias[i])
    xfmlp.sendMat(mat_B0)
    xfmlp.addGEMMOp (mat_A[0], mat_B0, mat_C[0], mat_bias[0], post_scale[0], post_scale[1])    
    xfmlp.addGEMMOp (mat_A[1], mat_C[0], mat_C[1], mat_bias[1], post_scale[0], post_scale[1]) 
    xfmlp.addGEMMOp (mat_A[2], mat_C[1], mat_C[2], mat_bias[2], post_scale[0], post_scale[1]) 
    xfmlp.addGEMMOp (mat_A[3], mat_C[2], mat_C[3], mat_bias[3], post_scale[0], post_scale[1])
    timePointKernel.append(time.time()) # send to FPGA
    xfmlp.execute()
    xfmlp.clearInstrBuf()
    timePointKernel.append(time.time()) # call kernel
    xfmlp.getMat(mat_C[0])  
    xfmlp.getMat(mat_C[1]) 
    xfmlp.getMat(mat_C[2]) 
    xfmlp.getMat(mat_C[3]) 
    timePointKernel.append(time.time()) # copy from FPGA
    freq = xfmlp.getFreq()
    test.test_perf(timePointKernel,total_operations,total_parallel_operations,freq,0,0,0)
    test.multiply_and_cmp(mat_C[3], mat_A[3], mat_C[2], mat_bias[3], m_size[3], n_size[3], post_scale)

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test=GemmTest()
  args, xclbin_opts = xfmlp.processCommandLine()
  xfmlp.createGEMMHandle(args, xclbin_opts)
  
  for PE in range(int(xclbin_opts["GEMX_numKernels"])):
      test.test_basic_randint( PE, 512, 512, 128, [16,17])
      test.test_basic_randint( PE, 256, 512, 128, [2,18])
      test.test_basic_randint( PE, 2048, 512, 128, [4,18])
      test.test_basic_randint( PE, 2048, 512, 128, [128,17])

  # test.test_rand_basic (32764, 0, 5, [1,0]) # larger matrix size will lead to hw timeout error in regression test
  test_multiInstrv1(32764, 512, 512, 128, True) 
  
