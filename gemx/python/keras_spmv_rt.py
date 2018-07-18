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

import gemx                                           
import numpy as np                                    
import math                                           
import scipy.sparse as sp                             
from gemx_rt import GemxRT                            
                             
class KerasSpmvRT(GemxRT):                                                                                                                    
    def __init__(self, keras_model, batch_sz, wgt_scale, xclbin_opt):                                                                         
        self.min_m = int(xclbin_opt["GEMX_spmvUramGroups"]) * int(xclbin_opt["GEMX_ddrWidth"])                                                
        self.min_k = int(xclbin_opt["GEMX_ddrWidth"])                                                                                         
        self._qw = keras_model.get_weights()[0::2]                                                                                              
        self._qb = keras_model.get_weights()[1::2]                                                                                              
        self._qw = [ np.float32(a*b) for a,b in zip(self._qw, wgt_scale)]                                                                         
        self._qb = [ np.float32(a*b) for a,b in zip(self._qb, wgt_scale)]                                                                         
        self.A_list = []                                                                                                                      
        self.sizes = []                                                                                                                       
        for i,wi in enumerate(self._qw):                                                                                                        
            wi = np.transpose(wi)                                                                                                             
            size_r,size_c,size_nnz,row,col,nnz = self.format_for_sparse_fpga(wi,self._qw[0].shape)                                              
            self.sizes.append((size_r,size_c,size_nnz))                                                                                       
            self.A_list.append(gemx.sendSpMat(row,col,nnz,self.min_k,np.float32))                                                                                                                                                                              
        self.out_dim = ( batch_sz, keras_model.layers[-1].output_shape[1] )                                                                   
        self.kmodel = keras_model

    def format_for_sparse_fpga ( self, weight_mat,shape):
        size_nnz = np.count_nonzero(weight_mat)
        row_size = max(shape[0],shape[1])
        col_size = max(shape[0],shape[1])
        m_index = np.nonzero(weight_mat)
        m_row = (m_index[0]).astype(np.int32)
        m_col = (m_index[1]).astype(np.int32)
        m_value = (weight_mat[m_row,m_col]).astype(np.float32)
        while size_nnz % self.min_k !=0:
            m_row = (np.append(m_row,0)).astype(np.int32)
            m_col = (np.append(m_col,0)).astype(np.int32)
            m_value = (np.append(m_value,0)).astype(np.float32)
            size_nnz = size_nnz + 1
        row_size_padded,col_size_padded = self.get_padded_shape([row_size,col_size], self.min_m, self.min_m)
        return row_size_padded,col_size_padded,size_nnz,m_row,m_col,m_value

    def predict ( self, inp, in_scale):
        C_list = [[]] * (len(self.kmodel.layers)+1)
        inp= self.format_for_fpga(inp, self.min_m,self.min_m)
        C_list[0] = np.transpose(inp * in_scale)
        for i,bi in enumerate(self._qb):
            bi = bi.reshape(bi.shape[0], 1)
            bi = self.format_for_fpga(bi, C_list[i].shape[1], self.sizes[i][0])
            for j in range(C_list[i].shape[1]):
                B = (C_list[i][:,j]).astype(np.float32)
                C = np.zeros ((self.sizes[i][0], 1), dtype=np.float32)
                gemx.sendMat(B)
                gemx.sendMat(C)
                gemx.addSPMVOp(self.A_list[i],B,C,self.sizes[i][2])
                gemx.execute()
                gemx.clearInstrBuf()
                gemx.getMat(C)
                if j == 0:
                    C_list[i+1] = C
                else:
                    C_list[i+1] = np.append(C_list[i+1],C,axis=1)
            C_list[i+1] = C_list[i+1] + np.transpose(bi)
        result = np.transpose(C_list[-1])
        return result[:self.out_dim[0],:self.out_dim[1]]
