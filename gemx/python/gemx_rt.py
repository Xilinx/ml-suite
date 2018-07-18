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

class GemxRT():
    def __init__(self, xclbin_opt, wgt,bias, wgt_scale, post_scale):
        
        #Ensuring min_m and min_n never fall below min_k is needed when chaining multiple GEMM operations
        #If min_m/min_n is less than min_k, using the output of a GEMM call where either dimension 
        #is less than min_k would lead to bad results if it's directly used as input for another GEMM operation  
        self.min_m = 32 * max (int(xclbin_opt["GEMX_gemmKBlocks"]), int(xclbin_opt["GEMX_gemmMBlocks"]) )
        self.min_k = 32 * int(xclbin_opt["GEMX_gemmKBlocks"])
        self.min_n = 32 * max ( int(xclbin_opt["GEMX_gemmKBlocks"]), int(xclbin_opt["GEMX_gemmNBlocks"]) ) 
        if type (wgt) != list:
            wgt = [wgt]
        
        if type(bias) != list:
            bias = [bias]
            
        self._wshape = []
        for w in wgt:
            self._wshape.append(w.shape)
            
        self._qw = [ np.int16(a*b) for a,b in zip(wgt, wgt_scale)]
        self._qb = [ np.int32(a*b) for a,b in zip(bias, wgt_scale)]
        for i,b in enumerate(self._qw):
            self._qw[i] = self.format_for_fpga( b, self.min_k, self.min_n)
            gemx.sendMat(self._qw[i])
            
        #in_row, in_col = self.get_padded_shape(in_dim, self.min_m, self.min_k)
        self.fpga_buf = []
        self.out_dim = None
        self.post_scale = post_scale
        self.batch_sz = 0
        
    def get_padded_shape ( self, shape, min_row, min_col):
        row_padded = int( math.ceil( np.float32(shape[0]) / min_row ) * min_row ) 
        col_padded = int( math.ceil( np.float32(shape[1]) / min_col ) * min_col )
        return row_padded,col_padded

    def format_for_fpga ( self, nparr, min_row, min_col):
        row_padded, col_padded = self.get_padded_shape ( nparr.shape, min_row, min_col)
        padded_arr = np.zeros ( (row_padded, col_padded), dtype=nparr.dtype, order='C')
        padded_arr[0:nparr.shape[0], 0:nparr.shape[1]] = nparr
        return padded_arr            
    
    def format_bias (self, b, dim, min_row, min_col):
        if b.ndim == 1:
            b = np.broadcast_to(b, dim )
            
        b = self.format_for_fpga( b, min_row, min_col)
        gemx.sendMat(b)    
        return b
    
    def init_fpgabuf (self, in_shape ):
        if self.batch_sz != in_shape[0]:
            self.batch_sz = in_shape[0]
            fpga_buf = []
            buf_dim = [in_shape]
        
            for i in self._wshape:
                buf_dim.append( (buf_dim[-1][0], i[1]) )
                
            self.out_dim = buf_dim[-1] 
                
            for d in buf_dim:
                d_padded = self.get_padded_shape(d, self.min_m, self.min_k)
                fpga_buf.append ( gemx.create_fpga_buf( d_padded, self._qw[0].dtype ) )
            
            self.fpga_buf = fpga_buf
            
            formatted_bias = []
            for dim,b  in zip (buf_dim[1:], self._qb):
                b = self.format_bias (b, dim, self.min_m, self.min_n)
                formatted_bias.append(b)   
            
            self._qb = formatted_bias           
    
    def loadInstr(self):
        gemx.clearInstrBuf()
        for i,(w_i,b_i) in enumerate( zip( self._qw, self._qb) ):
            gemx.addGEMMOp( self.fpga_buf[i], w_i , self.fpga_buf[i+1], b_i, self.post_scale[i][0], self.post_scale[i][1])
            
    def predict ( self, inp, in_scale):
        self.init_fpgabuf(inp.shape)
        self.loadInstr()
        
        padded_arr = self.format_for_fpga(inp*in_scale, self.min_m, self.min_k)
        
        #print ("input shape", padded_arr.shape)
        np.copyto(self.fpga_buf[0], np.int16( padded_arr ), casting='same_kind', where=True)
        gemx.sendMat(self.fpga_buf[0])
        gemx.execute()
        gemx.getMat (self.fpga_buf[-1])
        return self.fpga_buf[-1][:self.out_dim[0],:self.out_dim[1]]                
    