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
from gemx_rt import GemxRT
        
class KerasRT(GemxRT):
    def __init__(self, keras_model, xclbin_opt, wgt_scale, post_scale):
        keras_w = keras_model.get_weights()[0::2]
        keras_b = keras_model.get_weights()[1::2]
        GemxRT.__init__(self, xclbin_opt, keras_w, keras_b, wgt_scale, post_scale)
        self.kmodel = keras_model
       
    def loadInstr(self):
        gemx.clearInstrBuf()

        for i,l in enumerate(self.kmodel.layers):
            act = l.get_config()['activation']
            if act == 'relu':
                gemx.addFCNOp( self.fpga_buf[i], self._qw[i], self.fpga_buf[i+1], self._qb[i], self.post_scale[i][0], self.post_scale[i][1], 0, 0)
            else:
                gemx.addGEMMOp( self.fpga_buf[i], self._qw[i], self.fpga_buf[i+1], self._qb[i], self.post_scale[i][0], self.post_scale[i][1])