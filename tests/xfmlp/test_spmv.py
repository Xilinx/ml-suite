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
import math
import test
import scipy.io as sio
import scipy.sparse as sp
from test import SpmvTest

xclbin_opts = [] # config data read from config_info.dat

def get_padded_size (size, min_size):
  size_padded = int( math.ceil( np.float32(size) / min_size ) * min_size ) 
  return size_padded

def common_spmv(row,col,data,m,k,nnz,vector_range):
  if xclbin_opts["GEMX_dataType"] == "float":
    data_type = np.float32
  elif xclbin_opts["GEMX_dataType"] == "int32_t":
    data_type = np.int32
  else:
     raise TypeError("type", xclbin_opts["GEMX_dataType"], "not supported") 
  ddrWidth = int(xclbin_opts["GEMX_ddrWidth"])
  min_k = ddrWidth
  if xclbin_opts["GEMX_useURAM"] == "1":
       min_nnz = ddrWidth
       min_m = ddrWidth * int(xclbin_opts["GEMX_spmvUramGroups"])
  else:
       spmvWidth = int(xclbin_opts["GEMX_spmvWidth"])
       min_nnz = spmvWidth
       min_m = spmvWidth * int(xclbin_opts["GEMX_spmvMacGroups"])
  while nnz%min_nnz !=0:
    # pad with 0s and adjust dimensions when necessary
    row = (np.append(row,0)).astype(np.int32)
    col = (np.append(col,0)).astype(np.int32)
    data = (np.append(data,0)).astype(np.float32)
    nnz = nnz + 1
  m = get_padded_size (m, min_m)
  k = get_padded_size (k, min_k)
  print ("size:",m,k,"nnz:",nnz)  
  if data_type == np.int32:
     B = np.random.randint(low=-vector_range, high=vector_range, size=(k, 1), dtype=np.int32)
  else:
     B = np.zeros ((k, 1), dtype=np.float32)
     test.fillMod(B,k,vector_range)
  C = np.zeros ((m, 1), dtype=data_type)
  A = xfmlp.sendSpMat(row,col,data,m,k,nnz,xclbin_opts,data_type)  
  xfmlp.sendMat(B)
  xfmlp.sendMat(C)
  xfmlp.addSPMVOp(A,B,C,nnz,xclbin_opts)
  xfmlp.execute()
  xfmlp.clearInstrBuf()
  xfmlp.getMat(C)
  test.multiply_and_cmp_spmv(row,col,data,m,k,nnz,B,C)

def test_spmv_mtxfile(mtxpath,vector_range):
  matA = sio.mmread(mtxpath)
  if sp.issparse(matA):
     row = (matA.row).astype(np.int32)
     col = (matA.col).astype(np.int32)
     data = (matA.data).astype(np.float32)
     m,k = matA.shape
     nnz = matA.nnz         
     common_spmv(row,col,data,m,k,nnz,vector_range)
  else:
     print ("only sparse matrix is supported")

def test_spmv_random(m,k,nnz,vector_range=32764):
  row  = np.random.randint(low=0, high=m, size=(nnz, 1), dtype=np.int32)
  col  = np.random.randint(low=0, high=k, size=(nnz, 1), dtype=np.int32)
  data = np.zeros ((nnz, 1), dtype=np.float32)
  nnz_min = random.randint(-vector_range, vector_range)
  for i in range(nnz):
     nnz_min += 0.3
     data[i,0] = nnz_min
  common_spmv(row,col,data,m,k,nnz,vector_range) 

if __name__ == '__main__':
  np.random.seed(123)  # for reproducibility
  test = SpmvTest()
  args, xclbin_opts = xfmlp.processCommandLine()
  
  if xclbin_opts["GEMX_useURAM"] == "1":
    xfmlp.createSPMVHandle(args, xclbin_opts)
  else:
    xfmlp.createSPMVBRAMHandle(args, xclbin_opts)
   
  #mtx file must be in Matrix Market format
  #test_spmv_mtxfile("./data/spmv/mario001.mtx",32764) 
  #test_spmv_mtxfile("./data/spmv/image_interp.mtx",32764) 
  #test_spmv_mtxfile("./data/spmv/raefsky3.mtx",32764) 
  #test_spmv_mtxfile("./data/spmv/stomach.mtx",32764)  
  #test_spmv_mtxfile("./data/spmv/torso3.mtx",32764)  
  
  test_spmv_random(96,128,256,32764)
  test_spmv_random(65472,65472,500000,32764) 
  test_spmv_random(12800,12800,1400000,32764) 
  xfmlp.printStats()
  
