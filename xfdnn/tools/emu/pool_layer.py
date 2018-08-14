##################################################
# Copyright 2018 Xilinx Inc.
##################################################
# The information disclosed to you hereunder (the "Materials") is provided solely for the selection and use of Xilinx products. To the
# maximum extent permitted by applicable law: (1) Materials are made available "AS IS" and with all faults, Xilinx hereby DISCLAIMS ALL
# WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
# MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in
# contract or tort, including negligence, or under any other theory of liability) for any loss or damage of any kind or nature related to,
# arising under, or in connection with, the Materials (including your use of the Materials), including for any direct, indirect, special,
# incidental, or consequential loss or damage (including loss of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was reasonably foreseeable or Xilinx had been advised of the
# possibility of the same. Xilinx assumes no obligation to correct any errors contained in the Materials or to notify you of updates to the
# Materials or to product specifications. You may not reproduce, modify, distribute, or publicly display the Materials without prior written
# consent. Certain products are subject to the terms and conditions of Xilinx's limited warranty, please refer to Xilinx's Terms of Sale which
# can be viewed at http://www.xilinx.com/legal.htm#tos; IP cores may be subject to warranty and support terms contained in a license
# issued to you by Xilinx. Xilinx products are not designed or intended to be fail-safe or for use in any application requiring fail-safe
# performance; you assume sole risk and liability for use of Xilinx products in such critical applications, please refer to Xilinx's Terms of
# Sale which can be viewed at http://www.xilinx.com/legal.htm#tos.
##################################################

import numpy as np
import math
import layer
import util

class pool_layer(layer.layer) :
    def __init__(self, pool_type = 'MAX', dim = None, stride = None, padding = False, mode = 'NHWC') :
        self.pool_type = pool_type
        self.pool_stride = stride
        self.pool_kernel_dim = dim
        self.pool_pad = padding
        self.mode = mode
        self.pad = None

    def prepare_layer(self, node, inps, variables) :
        if node.type == 'MaxPool' :
            self.pool_type = 'MAX'
        elif node.type == 'AvgPool' :
            self.pool_type = 'AVG'
        self.pool_kernel_dim = node.get_attr('ksize')
        self.pool_stride = node.get_attr('strides')
        self.pool_pad = (node.get_attr('padding') == 'SAME')
        self.setInput(inps)
        self.setOutput(node.name)
        self.shape = node.outputs[0].shape
        return self
        
        
    def set_params(self, layer_params, variables) :
        if layer_params.pool == 1 :
            self.pool_type = 'AVG'
        stride = layer_params.strides
        kernel = layer_params.kernel_sizes
        self.pool_kernel_dim = [kernel.batches, kernel.channels, kernel.height, kernel.width]
        self.pool_stride = [stride.batches, stride.channels, stride.height, stride.width]
        if layer_params.tf_pad.lower() == 'valid' :
            self.pad = layer_params.paddings
        self.pool_pad = (layer_params.tf_pad.lower() == 'same')
        self.setInput(layer_params.bottoms)
        self.setOutput(layer_params.name)
        return self

    def forward_exec(self, inputs) :
        # [aaronn] satyakee: we need to get padding config from pydot
        #print "ANDBG pool stride %s" % str(self.pool_stride)
        #print "ANDBG pool kdim %s" % str(self.pool_kernel_dim)
        #if self.pool_type != 'AVG':
        #  self.pool_kernel_pad = True # XXX FIXME
        #print "ANDBG pool kpad %s" % str(self.pool_kernel_pad)

        inp = inputs[0]
        #if self.pool_type != 'AVG':
          #self.pool_pad = True # XXX FIXME
        # print "ANDBG pool kpad %s" % str(self.pool_kernel_pad)
        print(inp.shape)
        #print "ANDBG pool_layer %s" % self.mode
        #print "ANDBG kernel %s" % str(self.pool_kernel_dim)
        if self.pool_pad or self.pad != None:
            inp = util.Pad_tf(inp, self.pool_kernel_dim, self.pool_stride, self.mode, ispool=True, pad_vals=self.pad)
        print(inp.shape)
        res = []
        if self.mode == 'NHWC' :
            for i in range(inp.shape[0]) :
                res.append([self.pool_reduce(inp[i])])
            res = np.concatenate(res)
            return res
        else :
            for i in range(inp.shape[0]) :
                res.append([self.pool_reduce_nchw(inp[i])])
            res = np.concatenate(res)
            return res

    def pool_reduce_nchw(self, inp_) :
        inp_arr = []
        ker = self.pool_kernel_dim
        ker_len = np.prod(ker)
        for i in range(0, inp_.shape[1] - self.pool_kernel_dim[2] + 1, self.pool_stride[2]) :
            for j in range(0, inp_.shape[2] - self.pool_kernel_dim[3] + 1, self.pool_stride[3]) :
                inp_arr.append(np.reshape(inp_[:, i:i+self.pool_kernel_dim[2], j:j+self.pool_kernel_dim[3]], [inp_.shape[0], ker_len]))
        inp_arr = np.array(inp_arr)
        if self.pool_type == 'MAX' :
            inp_arr = np.amax(inp_arr, axis = 2)
        elif self.pool_type == 'AVG' :
            inp_arr = np.mean(inp_arr, axis = 2)
        elif self.pool_type == 'RMS' :
            inp_arr = np.mean(np.square(inp_arr), axis = 2)
        inp_arr = np.transpose(inp_arr)
        res_shape = [inp_.shape[0], int(math.floor((inp_.shape[1] - ker[2])/self.pool_stride[2])+1), int(math.floor((inp_.shape[2] - ker[3])/self.pool_stride[3])+1)]
        #print res_shape
        return np.reshape(inp_arr, res_shape)
    
    def pool_reduce(self, inp_) :
        inp_arr = []
        ker = self.pool_kernel_dim
        ker_len = np.prod(ker)
        for i in range(0, inp_.shape[0] - self.pool_kernel_dim[1] + 1, self.pool_stride[1]) :
            for j in range(0, inp_.shape[1] - self.pool_kernel_dim[2] + 1, self.pool_stride[2]) :
                inp_arr.append(np.reshape(inp_[i:i+self.pool_kernel_dim[1], j:j+self.pool_kernel_dim[2]], [ker_len, inp_.shape[2]]))
        inp_arr = np.array(inp_arr)
        if self.pool_type == 'MAX' :
            inp_arr = np.amax(inp_arr, axis = 1)
        elif self.pool_type == 'AVG' :
            inp_arr = np.mean(inp_arr, axis = 1)
        elif self.pool_type == 'RMS' :
            inp_arr = np.mean(np.square(inp_arr), axis = 1)
        res_shape = [int(math.ceil((inp_.shape[0] - ker[1] + 1)/float(self.pool_stride[1]))), int(math.ceil((inp_.shape[1] - ker[2] + 1)/float(self.pool_stride[2]))), inp_.shape[2]]
        return np.reshape(inp_arr, res_shape)

        
