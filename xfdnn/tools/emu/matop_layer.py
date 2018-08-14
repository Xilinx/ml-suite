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

class matop_layer(layer.layer) :
  def __init__(self, optype, weights=None, Bias = None, activation = None) :
    self.optype = optype
    self.weights = weights
    self.Bias = Bias
    self.activation = activation

  def prepare_layer(self, node, inps, variables) :
    if self.optype == 'MatMul' or self.optype == 'InnerProduct' :
      self.weights = variables[inps[1]]
      if len(inps) == 3 :
        self.Bias = variables[inps[2]]
      self.setInput(inps[:1])
    elif self.optype == 'BiasAdd' :
      self.Bias = variables[inps[1]]
      self.setInput(inps[:1])
    else :
        self.setInput(inps)
    self.setOutput(node.name)
    self.shape = node.outputs[0].shape
    print(self.inputs)
    return self

  def set_params(self, layer_params, variables) :
    print(layer_params.layer[0].inputs)
    if self.optype == 'MatMul' and layer_params.type[0] == 'InnerProduct' :
      varName = layer_params.name + "_weights" 
      self.weights = np.copy(layer_params.data.weights)
      variables[varName] = self.weights
      self.setInput(layer_params.bottoms[:1])
      self.setOutput(layer_params.name)
    elif self.optype == 'BiasAdd' :
      varName = layer_params.name+'_bias'
      self.Bias = np.copy(layer_params.data)
      variables[varName] = self.Bias
      self.setInput(layer_params.bottoms[:1])
      self.setOutput(layer_params.name)
    else : 
      self.setInput(layer_params.bottoms)
      self.setOutput(layer_params.name)
    if layer_params.relu :
        self.activation = 'ReLU'
    return self

  def forward_exec(self, inputs) :
    res = None
    if len(inputs) > 1 :
      if self.optype == 'Add' :
        res = np.add(inputs[0], inputs[1])
      elif self.optype == 'Sub' :
        res = np.subtract(inputs[0], inputs[1])
    elif self.optype == 'MatMul' or self.optype=='Inner':
      if inputs[0].shape[1] == self.weights.shape[0] :
        res = np.dot(inputs[0], self.weights)
      else :
        res = []
        for i in range(inputs[0].shape[0]) :
            res.append(np.dot(self.weights, inputs[0][i]))
        res = np.array(res)
      if type(self.Bias) != type(None) :
        res = np.add(res, self.Bias)
    elif self.optype == 'BiasAdd' or self.optype == 'Add' :
      res = np.add(inputs[0], self.Bias)
    if self.activation == 'ReLU' :
        res = util.ReLU(res)
    return res
