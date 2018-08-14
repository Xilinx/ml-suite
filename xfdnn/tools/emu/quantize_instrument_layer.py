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
import layer
import timeit
import xfdnn.tools.quantize.quantize_base as quant

class quantize_instrument_conv_layer(layer.layer) :
  # layer expects 2 inputs:
  # orig layer's input, output

  def __init__(self, quantizer, bitwidth, quantize_keys, weights=None):
    self.quantizer = quantizer
    self.bitwidth = bitwidth
    self.quantize_prev_key = quantize_keys[0]
    self.quantize_key = quantize_keys[1]
    self.filter_weights = weights

  def set_params(self, layer_params, variables) :
    self.setInput(layer_params.bottoms)
    self.setOutput(layer_params.name)

  def forward_exec(self, inputs):
    inBlob = inputs[0]
    outBlob = inputs[1]

    self.quantizer.bw_layer_in[self.quantize_key] = self.bitwidth
    self.quantizer.bw_layer_out[self.quantize_key] = self.bitwidth

    # input
    if self.quantize_prev_key in self.quantizer.th_layer_out:
      threshold = self.quantizer.th_layer_out[self.quantize_prev_key]
    else:
      threshold = quant.ThresholdLayerOutputs(inBlob, self.bitwidth)
    self.quantizer.th_layer_in[self.quantize_key] = threshold

    # output
    threshold = quant.ThresholdLayerOutputs(outBlob, self.bitwidth)
    self.quantizer.th_layer_out[self.quantize_key] = threshold

    # weights
    qdata = self.filter_weights.transpose(3,2,0,1) 
    threshold = quant.ThresholdWeights(qdata, self.bitwidth)
    self.quantizer.bw_params[self.quantize_key] = self.bitwidth 
    self.quantizer.th_params[self.quantize_key] = threshold

    return 0

class quantize_instrument_concat_layer(layer.layer) :
  def __init__(self, quantizer, bitwidth, quantize_keys):
    self.quantizer = quantizer
    self.quantize_keys = quantize_keys
    self.bitwidth = bitwidth

  def set_params(self, layer_params, variables) :
    self.setInput(layer_params.bottoms)
    self.setOutput(layer_params.name)

  def forward_exec(self, inputs):
    inputBlobs = inputs[:-1]
    outputBlob = inputs[-1]

    bitwidth = self.bitwidth
    threshold = quant.ThresholdLayerOutputs(outputBlob, bitwidth)

    thisLayerName = self.quantize_keys[-1]
    self.quantizer.bw_layer_out[thisLayerName] = bitwidth
    self.quantizer.th_layer_out[thisLayerName] = threshold
    self.quantizer.bw_layer_in[thisLayerName] = bitwidth
    self.quantizer.th_layer_in[thisLayerName] = threshold

    # propagate/sync thresholds to all concat inputs
    for prevLayerName in self.quantize_keys[:-1]:
      self.quantizer.bw_layer_out[prevLayerName] = bitwidth
      self.quantizer.th_layer_out[prevLayerName] = threshold

    return 0
