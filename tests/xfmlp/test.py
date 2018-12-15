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
import scipy.sparse as sp
import subprocess

# test.py includes all the common test function shared by gemm, fcn and spmv engine
class Test:
  def cmp(self,A, B):
      if np.array_equal(A, B):
          print ("Success!\n")
      else:
          print ("not equal :(")
          sys.exit()
          
  def cmpWithinTolerance(self,A, B):
      if np.allclose(A, B,1e-3,1e-5):
          print ("Success!\n")
      else:
          print (A.shape, B.shape)
          np.savetxt("C.np", A, fmt="%f")
          np.savetxt("C_cpu.np", B, fmt="%f")
          diff = np.isclose(A, B,1e-3,1e-5)
          countDiff = diff.shape[0] - np.count_nonzero(diff)
          print ("not equal, number of mismatches = ", countDiff)
          mismatch = ((diff==0).nonzero())
          print ("mismatches are in ",mismatch[0])
          for i in mismatch[0]:
            print (A[i]," is different from ",B[i])         
          sys.exit()  
          
  def multiply_and_cmp(self,C, A, B, X, m, n, post_scale, pRelu_val = [1,0]):
      # Calculate golden C
      #start_compute = time.time()
      m64 = np.int64(np.round(np.matmul(np.float64(A), np.float64(B))))  # intermediate accumulation to 64 bits
      #print ("float64 compute elapsed:", time.time() - start_compute)
      #m64 = np.matmul(np.int64(A), np.int64(B)) # intermediate accumulation to 64 bits
      bias64 = np.int64(X)  # bias to 64 bits
      output64 = m64 + bias64
      o64d = output64 * post_scale[0]
      o64m = o64d // (2 ** post_scale[1])
      o64m = np.int16(o64m)
      if pRelu_val != [1,0]:
        for entry in np.nditer(o64m, op_flags=['readwrite']):
          if entry < 0:
              entry[...] = entry * pRelu_val[0] // (2 ** pRelu_val[1])
      C_cpu = np.int16(o64m)  # scale down for 16 bits    
      if np.array_equal(C, C_cpu):
          print ("Success!\n")
      else:
          print ("Not equal!")
          print (C.shape, C_cpu.shape)
          np.savetxt("cpu_out.np", C_cpu, fmt="%d")
          np.savetxt("fpga_out.np", C, fmt="%d")
          np.savetxt("bias.np", X, fmt="%d")
          np.savetxt("A.np", A, fmt="%d")
          np.savetxt("B.np", B, fmt="%d")
          sys.exit()    
  
  def check (self, o, golden ):
      if np.array_equal(o, golden):
          print ("Success!\n")
      else:
          print ("Not equal!")
          print (o.shape, golden.shape)
          np.savetxt("cpu_out.np", C_cpu, fmt="%d")
          np.savetxt("fpga_out.np", C, fmt="%d")
          #np.savetxt("bias.np", X, fmt="%d")
          #np.savetxt("A.np", A, fmt="%d")
          #np.savetxt("B.np", B, fmt="%d")         
          sys.exit()   
          
  def gen_rand_dim (self, min_mult, max):
      rand_dim = random.randint(1, int(max/min_mult))
      return rand_dim * min_mult
  
  def gen_rand_matrix (self, dtype, row, col):
      max_val = np.iinfo(dtype).max
      min_val = np.iinfo(dtype).min
      return np.random.randint(low=min_val, high=max_val, size=(row, col), dtype=dtype)
      
  def test_basic_randint (self,PE, xclbin_opts, post_scale, max_dim):
      rand_m = self.gen_rand_dim ( 32 * int(xclbin_opts["GEMX_gemmMBlocks"]), max_dim )
      rand_k = self.gen_rand_dim ( 32 * int(xclbin_opts["GEMX_gemmKBlocks"]), max_dim )
      rand_n = self.gen_rand_dim ( 32 * int(xclbin_opts["GEMX_gemmNBlocks"]), max_dim )
      mat_A = self.gen_rand_matrix ( np.int16, rand_m, rand_k)
      mat_B = self.gen_rand_matrix ( np.int16, rand_k, rand_n)
      bias = self.gen_rand_matrix ( np.int32, rand_m, rand_n)
      
      self.test_basic(PE,mat_A, mat_B, bias, post_scale)
      
  def test_basic(self,PE, mat_A, mat_B, bias, post_scale = [1,1]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test_basic(PE=%d): %d %d %d %d %d" % (PE,m, k, n, post_scale[0], post_scale[1] )) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros( (m, n), dtype=np.int16)
      xfmlp.sendMat(mat_A,PE)
      xfmlp.sendMat(mat_B,PE)
      xfmlp.sendMat(C_fpga,PE)    
      xfmlp.sendMat(bias, PE)
      xfmlp.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], PE) # default test_basic will call addGEMMOp
      xfmlp.execute(PE)
      xfmlp.clearInstrBuf(PE)
      xfmlp.getMat(C_fpga,PE)
      self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
   
  def test_perf(self,timePointKernel, total_operations, total_parallel_operations, freq, m, k, n):
      Execute_Time = (timePointKernel[2] - timePointKernel[1])*1e3
      API_Time = (timePointKernel[3] - timePointKernel[0])*1e3
      timeMsAt100pctEff = total_parallel_operations / 2 / 32 / 32 / ( freq * 1e6 ) * 1e3
      effKernelPct = 100 * timeMsAt100pctEff / Execute_Time
      effApiPct = 100 * timeMsAt100pctEff / API_Time
      perfKernelInTops = total_operations / (Execute_Time * 1e-3) / 1e12
      perfApiInTops = total_operations/ (API_Time * 1e-3) / 1e12;
      print ("DATA_CSV:DdrWidth,Freq,M,K,N,Ops,TimeKernelMs,TimeApiMs,EffKernelPct,EffApiPct,PerfKernelTops,PerfApiTops")
      print ("DATA_CSV:32,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f" % (freq,m,k,n,total_operations,Execute_Time,API_Time,effKernelPct,effApiPct,perfKernelInTops,perfApiInTops))
  
  def check_input(self, m_size, k_size, n_size, xclbin_opts):
      m_block = int(xclbin_opts["GEMX_gemmMBlocks"])
      k_block = int(xclbin_opts["GEMX_gemmKBlocks"])
      n_block = int(xclbin_opts["GEMX_gemmNBlocks"])
      ddr_width = int(xclbin_opts["xfmlp_ddrWidth"])
      if m_size%(m_block*ddr_width) !=0:
         print ("m must be multiple of", m_block, "and", ddr_width)
         sys.exit()
      elif k_size%(k_block*ddr_width) !=0:
         print ("k must be multiple of", k_block, "and", ddr_width)
         sys.exit()
      elif n_size%(n_block*ddr_width) !=0:
         print ("n must be multiple of", n_block, "and", ddr_width)  
         sys.exit()
         
  def test_textfiles(self, path_to_a, path_to_b, path_to_bias, post_scale):        
      mat_A = np.loadtxt(path_to_a, dtype=np.int16)
      mat_B = np.loadtxt(path_to_b, dtype=np.int16)
      bias = np.loadtxt(path_to_bias, dtype=np.int32)
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
      xfmlp.sendMat(mat_A)
      xfmlp.sendMat(mat_B)
      xfmlp.sendMat(C_fpga)    
      xfmlp.sendMat(bias)
      xfmlp.addGEMMOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1])
      xfmlp.execute()
      xfmlp.clearInstrBuf()
      xfmlp.getMat(C_fpga)  
      self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
      
  def get_freq(self, command): 
      #command could be $XILINX_OPENCL/runtime/bin/xbsak query -d BOARD_ID or $XILINX_XRT/bin/xbutil query -d BOARD_ID
      nextLine_isFreq = False
      freq = 250 # when failed to get board frequency will use 250MHz for reporting
      proc = subprocess.check_output(command.split())
      for line in proc.splitlines():
        if nextLine_isFreq:
          freq = int(line.split()[1])
          break
        elif "OCL Frequency" in line:
          nextLine_isFreq = True
      print("when failed to get board frequency will use 250MHz for reporting")
      return freq
      
class FcnTest(Test):       
    
  def test_basic_randint (self,PE, xclbin_opts, post_scale, RELU_scale, max_dim):
      rand_m = self.gen_rand_dim ( 32 * int(xclbin_opts["GEMX_gemmMBlocks"]), max_dim )
      rand_k = self.gen_rand_dim ( 32 * int(xclbin_opts["GEMX_gemmKBlocks"]), max_dim )
      rand_n = self.gen_rand_dim ( 32 * int(xclbin_opts["GEMX_gemmNBlocks"]), max_dim )
      mat_A = self.gen_rand_matrix ( np.int16, rand_m, rand_k)
      mat_B = self.gen_rand_matrix ( np.int16, rand_k, rand_n)
      bias = self.gen_rand_matrix ( np.int32, rand_m, rand_n)      
      self.test_basic(PE,mat_A, mat_B, bias, post_scale, RELU_scale)    
      
  def test_basic(self,PE, mat_A, mat_B, bias, post_scale=[1, 1], RELU_scale = [1,0]):
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      print ("test Fcn")
      print ("test_basic: %d %d %d %d %d" % (m, k, n, post_scale[0], post_scale[1])) 
      print ("A: ", np.amax(mat_A), np.amin(mat_A), np.average(mat_A))
      print ("B: ", np.amax(mat_B), np.amin(mat_B), np.average(mat_B))
      print ("bias: ", np.amax(bias), np.amin(bias), np.average(bias))
      C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
      xfmlp.sendMat(mat_A, PE)
      xfmlp.sendMat(mat_B, PE)
      xfmlp.sendMat(C_fpga, PE)    
      xfmlp.sendMat(bias, PE)
      xfmlp.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], RELU_scale[0], RELU_scale[1], PE)
      xfmlp.execute(PE)
      xfmlp.clearInstrBuf(PE)
      xfmlp.getMat(C_fpga, PE)  
      self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale, RELU_scale)
      
  def test_textfiles(self, path_to_a, path_to_b, path_to_bias,post_scale):        
      mat_A = np.loadtxt(path_to_a, dtype=np.int16)
      mat_B = np.loadtxt(path_to_b, dtype=np.int16)
      bias = np.loadtxt(path_to_bias, dtype=np.int32)
      m = mat_A.shape[0]
      k = mat_A.shape[1]
      n = mat_B.shape[1]
      C_fpga = np.zeros((m, n), dtype=np.int16, order='C')
      xfmlp.sendMat(mat_A)
      xfmlp.sendMat(mat_B)
      xfmlp.sendMat(C_fpga)    
      xfmlp.sendMat(bias)
      xfmlp.addFCNOp (mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1], 1, 0)
      xfmlp.execute()
      xfmlp.clearInstrBuf()
      xfmlp.getMat(C_fpga)  
      self.multiply_and_cmp(C_fpga, mat_A, mat_B, bias, m, n, post_scale)
        
class SpmvTest(Test):
  def multiply_and_cmp_spmv(self,row,col,data,m,k,nnz,B,C):
      if B.dtype == np.int32:
        C_cpu = np.zeros ((m, 1), dtype=np.int32)
        data_cpu = np.zeros ((m, 1), dtype=np.int32)
        data_cpu = data.astype(np.int32)
      elif B.dtype == np.float32:
        C_cpu = np.zeros ((m, 1), dtype=np.float32)
        data_cpu = np.zeros ((m, 1), dtype=np.float32)
        data_cpu = data.astype(np.float32)
      else:
        raise TypeError("type", B.dtype, "not supported") 
      for i in range(nnz):
        C_cpu[row[i]] += B[col[i]] * data_cpu[i]
      self.cmpWithinTolerance(C, C_cpu)
      
  def fillMod(self,B,size,Max):
      l_val = 1.0
      l_step = 0.3
      l_drift = 0.00001
      l_sign = 1
      for i in range(size):
        B[i,0] = l_val
        l_val += l_sign * l_step
        l_step += l_drift
        l_sign = -l_sign;
        if l_val > Max:
          l_val -= Max
          
  def test_spmv_keras(self, len_layers, w_list, b_list, inp, shape):
      C_list = [[]] * (len_layers+1)                     
      C_list[0] = np.transpose(inp)                        
      for i in range(len_layers):                        
            wi = np.transpose(w_list[i])                                
            A = sp.csr_matrix(wi)                                       
            B = sp.csr_matrix(C_list[i])                                                    
            C = np.multiply(A,B)                                                             
            C = C.toarray()
            if i != len_layers -1:
              C = C + b_list[i]
              C = C.clip(min=0)                                                                    
            C_list[i+1] = C                                                                 
      result = np.transpose(C_list[-1])                                                   
      return result[:shape[0],:shape[1]] 

class GemmTest(Test):               
  pass
