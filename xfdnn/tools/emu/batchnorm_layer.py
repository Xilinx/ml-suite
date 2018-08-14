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

import layer
import util

import numpy as np
import copy

class batchnorm_layer(layer.layer) :
    def __init__(self, mean = 0, variance = 1, beta = 0, gamma = 1, var_ep = 0, mode = "NHWC", activation = None) :
        self.mean = mean
        self.var = variance
        self.beta = beta
        self.gamma = gamma
        self.var_ep = var_ep
        self.activation = activation

    def prepare_layer(self, node, inps, variables) :
        #print('bn beta', inps[3], node.inputs[3])
        self.mean = variables[inps[1]]
        self.var = variables[inps[2]]
        self.beta = variables[inps[3]]
        self.gamma = variables[inps[4]]
        self.var_ep = node.get_attr('variance_epsilon')
        self.setInput(inps[:1])
        self.shape = node.outputs[0].shape
        self.setOutput(node.name)
        return self

    def set_params(self, layer_params, variables) :
        #if layer_params.tops[0] == 'module_apply_default/resnet_v2_50/block1/unit_1/bottleneck_v2/conv2/BatchNorm/FusedBatchNorm' :
        print(layer_params) 
        self.mean = layer_params.data.mu
        self.var = layer_params.data.sigma_square
        if layer_params.scaling :
            scale_layer_params = layer_params.extras_and_future[0]
            self.gamma = scale_layer_params.data.gamma
            self.beta = scale_layer_params.data.beta
        if layer_params.relu :
            self.activation = 'ReLU'
        self.var_ep = layer_params.shapes[2]
        self.setInput(layer_params.bottoms)
        self.setOutput(layer_params.name)
        self.mode = "NCHW"
        return self

    def forward_exec(self, inputs) :
        print(('bn',inputs[0].shape, type(self.mean), type(self.var), self.gamma, self.beta, self.var_ep))
        if self.mode == "NCHW" :
            res = copy.deepcopy(inputs[0])
            for i in range(inputs[0].shape[1]) :
                res[:,i] = (res[:,i] - self.mean[i])/np.sqrt(self.var[i] + self.var_ep) 
                if self.gamma is not 1 :
                    res[:,i] = res[:,i]*self.gamma[i]
                if self.beta is not 0 :
                    res[:,i] = res[:,i] + self.beta[i]
            if self.activation == 'ReLU' :
                res = util.ReLU(res)
            return res
        else :
            bn = (inputs[0] - self.mean)/np.sqrt(self.var + self.var_ep)
            bn = bn*self.gamma + self.beta
            return bn
        
