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

class conv_layer(layer.layer) :
  def __init__(self, weights = None, stride = [1,1,1,1], activation = None, padding = False, biases = 0, mode = 'NHWC') :
    self.filter_weights = weights
    self.conv_stride = stride
    self.activation_fn = activation
    self.padding = padding
    self.biases = biases
    self.mode = mode
    self.pad = None

  def prepare_layer(self, node, inps, variables) :
    self.filter_weights = variables[inps[1]]
    self.conv_stride = node.get_attr('strides')
    self.padding = (node.get_attr('padding') == 'SAME')
    self.setInput(inps[:1])
    self.shape = node.outputs[0].shape
    self.setOutput(node.name)
    return self

  def set_params(self, layer_params, variables) :
    strides = layer_params.strides
    self.filter_weights = layer_params.data.weights
    if self.mode == 'NHWC':
      self.conv_stride = [strides.batches, strides.height,strides.width, strides.channels]
    else:
      self.conv_stride = [strides.batches, strides.channels, strides.height,strides.width]
    if type(layer_params.data.biases) != type(None) :
        self.biases = layer_params.data.biases
    if layer_params.tf_pad.lower() == 'valid' :
        self.pad = layer_params.paddings
    self.padding = (layer_params.tf_pad.lower() == 'same')
    if layer_params.relu :
        self.activation_fn = 'ReLU'
    self.setInput(layer_params.bottoms)
    self.setOutput(layer_params.name)
    return self

  def forward_exec(self,inputs) :
    #Assuming input is n, h, w, c
    inp = inputs[0]
    conv_out = []
    if self.padding or self.pad != None:
      inp = util.Pad_tf(inp, self.filter_weights.shape, self.conv_stride, self.mode, pad_vals=self.pad)
    print('conv', inp.shape)

    for i in range(inp.shape[0]) :
      conv = self.performConv(inp[i])
      conv = self.addBias(conv)
      conv_out.append([conv])

    conv_out = np.concatenate(conv_out)

    if self.activation_fn == 'ReLU' :
      conv_out = util.ReLU(conv_out)
    print(conv_out.shape)
    return conv_out
 
  def performConv_nchw(self, pad_pic):
    fil = self.filter_weights
    pic_arr = []
    n_cout = fil.shape[0]
    chw_filter = np.prod(fil.shape[1:])
    for i in range(0,pad_pic.shape[1]-fil.shape[2]+1, self.conv_stride[2]) :
        for j in range(0, pad_pic.shape[2]-fil.shape[3] + 1, self.conv_stride[3]) :
            pic_arr.append(np.reshape(\
              pad_pic[:,i:i+fil.shape[2],j:j+fil.shape[3]],[chw_filter]))
    pic_arr = np.array(pic_arr)
    fil_arr = np.reshape(self.filter_weights, [n_cout, chw_filter])
    res = np.dot(fil_arr, np.transpose(pic_arr))

    # [aaronn] hack to fix dot results. 
    # FIXME satyakee: please check 
    #res_shape = [n_cout, int(math.floor((pad_pic.shape[1] - fil.shape[2])/self.conv_stride[2]) + 1), int(math.floor((pad_pic.shape[2] - fil.shape[3])/self.conv_stride[3]) + 1)]
    res_shape = [n_cout, int(math.ceil((pad_pic.shape[1] - fil.shape[2] + 1)/float(self.conv_stride[2]))), int(math.ceil((pad_pic.shape[2] - fil.shape[3] + 1)/float(self.conv_stride[3])))]
    res = np.reshape(res, res_shape)
    return res
    
  def performConv(self, pad_pic) :
    if self.mode != 'NHWC':
      return self.performConv_nchw(pad_pic)

    fil = self.filter_weights
    pic_arr = []
    n_cout = fil.shape[3]
    chw_filter = np.prod(fil.shape[:3])
    for i in range(0,pad_pic.shape[0]-fil.shape[0]+1, self.conv_stride[1]) :
        for j in range(0, pad_pic.shape[1]-fil.shape[1] + 1, self.conv_stride[2]) :
            pic_arr.append(np.reshape(\
              pad_pic[i:i+fil.shape[0],j:j+fil.shape[1]],[chw_filter]))
    pic_arr = np.array(pic_arr)
    fil_arr = np.reshape(self.filter_weights, [chw_filter, n_cout])
    res = np.dot(pic_arr,fil_arr)
    res_shape = [int(math.ceil((pad_pic.shape[0] - fil.shape[0] + 1)/float(self.conv_stride[1]))), int(math.ceil((pad_pic.shape[1] - fil.shape[1] + 1)/float(self.conv_stride[2]))), n_cout]
    return np.reshape(res, res_shape)

  def addBias(self, pic):
    if type(self.biases) == np.ndarray or self.biases != 0 :
      if self.mode == 'NHWC':
        return pic + self.biases

      out = np.copy(pic) 
      for j in range(self.biases.shape[0]) :
        out[j] = np.add(out[j], self.biases[j])
      return out

    return pic
