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

import numpy as np

class reshape_layer(layer.layer) :
    def __init__(self, reshapedim = [0,1,2,3]) :
        self.new_dim = reshapedim

    def prepare_layer(self, node, inps, variables) :
        self.new_dim = variables[inps[1]]
        self.setInput(inps[:1])
        self.setOutput(node.name)
        self.shape = node.outputs[0].shape
        return self
        
    def set_params(self, layer_params, variables) :
        self.new_dim = layer_params.data
        self.setInput(layer_params.bottoms)
        self.setOutput(layer_params.name)
        return self

    def forward_exec(self, inputs) :
        inp_ = np.copy(inputs[0])
        out = []
        for i in range(len(inp_)) :
            out.append(np.reshape(np.array([inp_[i]]), self.new_dim))
        return np.concatenate(out)
