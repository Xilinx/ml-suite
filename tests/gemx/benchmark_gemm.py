##################################################
#Copyright (c) 2018, Xilinx, Inc.
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without modification,
#are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice,
#this list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#this list of conditions and the following disclaimer in the documentation
#and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its contributors
#may be used to endorse or promote products derived from this software
#without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
#IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
#HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
#EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################

import test
from test import GemmTest
import numpy as np
import gemx
import time
def test_perf_gemm_gemm(A_range, B_range, bias_range, m, k, n, post_scale):
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
    gemx.sendMat(mat_A)
    gemx.sendMat(mat_B)
    gemx.sendMat(C_fpga)    
    gemx.sendMat(bias)
    gemx.addGEMMOp ( mat_A, mat_B, C_fpga, bias, post_scale[0], post_scale[1])
    timePointKernel.append(time.time()) # send to FPGA
    gemx.execute()
    timePointKernel.append(time.time()) # call kernel
    gemx.getMat(C_fpga)  
    timePointKernel.append(time.time()) # copy from FPGA
    total_operations = 2 * m * n * k + m * n * 3
    total_parallel_operations = 2 * m * n * k
    freq = gemx.getFreq()
    test.test_perf(timePointKernel,total_operations,total_parallel_operations,freq,m,k,n)
    if m > 4096 and n > 4096 and k > 4096:
      print("Skip golden comparision because large matrix size")
    else:
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
      gemx.sendMat(mat_A[i])
      gemx.sendMat(mat_C[i])
      gemx.sendMat(mat_bias[i])
    gemx.sendMat(mat_B0)
    gemx.addGEMMOp (mat_A[0], mat_B0, mat_C[0], mat_bias[0], post_scale[0], post_scale[1])    
    gemx.addGEMMOp (mat_A[1], mat_C[0], mat_C[1], mat_bias[1], post_scale[0], post_scale[1]) 
    gemx.addGEMMOp (mat_A[2], mat_C[1], mat_C[2], mat_bias[2], post_scale[0], post_scale[1]) 
    gemx.addGEMMOp (mat_A[3], mat_C[2], mat_C[3], mat_bias[3], post_scale[0], post_scale[1])
    timePointKernel.append(time.time()) # send to FPGA
    gemx.execute()
    timePointKernel.append(time.time()) # call kernel
    gemx.getMat(mat_C[0])  
    gemx.getMat(mat_C[1]) 
    gemx.getMat(mat_C[2]) 
    gemx.getMat(mat_C[3]) 
    timePointKernel.append(time.time()) # copy from FPGA
    freq = gemx.getFreq()
    test.test_perf(timePointKernel,total_operations,total_parallel_operations,freq,0,0,0)
    if np.max(m_size) > 4096 and np.max(k_size) > 4096 and np.max(n_size) > 4096:
      print("Skip golden comparision because large matrix size")
    else:
      test.multiply_and_cmp(mat_C[3], mat_A[3], mat_C[2], mat_bias[3], m_size[3], n_size[3], post_scale)

if __name__ == '__main__':
    np.random.seed(123)  # for reproducibility
    test=GemmTest()
    parser = gemx.processCommandLine()
    args = parser.parse_args()    
    
    gemx.createGEMMHandle(args.xclbin, args.gemxlib, args.device, args.numKernel)
    m_size=np.array([512,512,2048,128])
    k_size=np.array([384,512,512,2048])
    n_size=np.array([128,128,128,128])   
    test_perf_multi_gemm(4, m_size, k_size, n_size, 32764, 32764, [1,0]) # run performance measurement
    gemx.printStats()
    
#    size = 256
#    while size < 16384:
#        test_perf(32764, 32764, 0, size, size, size, [1,0])
#        size = size * 2

     