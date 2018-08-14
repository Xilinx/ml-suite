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
import timeit

class quantize_layer(layer.layer) :
  def __init__(self, quantize_key, xdnn_env):
    self.quantize_key = quantize_key
    self.xdnn_env = xdnn_env

  def set_params(self, layer_params, variables) :
    self.setInput(layer_params.bottoms)
    self.setOutput(layer_params.name)

  def forward_exec(self, inputs) :
    startTime = timeit.default_timer()
    xdnnParams = self.xdnn_env.get_params()

    blob = np.copy(inputs[0])

    if xdnnParams['useGlobalScale']:
      return blob * xdnnParams['scaleB']

    blob = np.ascontiguousarray(blob, dtype=np.float32)
    qp = xdnnParams['quantDB'][self.quantize_key]
    xdnnParams['api'].quantizeTensor(\
      qp['th_layer_in'], qp['bw_params'], blob)

    elapsedTime = timeit.default_timer() - startTime
    #print "[time] quantize_layer: %.2fms" % (elapsedTime*1000)

    return blob

class unquantize_layer(layer.layer) :
  def __init__(self, quantize_key, xdnn_env):
    self.quantize_key = quantize_key
    self.xdnn_env = xdnn_env

  def set_params(self, layer_params, variables) :
    self.setInput(layer_params.bottoms)
    self.setOutput(layer_params.name)

  def forward_exec(self, inputs) :
    startTime = timeit.default_timer()
    xdnnParams = self.xdnn_env.get_params()

    blob = np.copy(inputs[0])
    if xdnnParams['useGlobalScale']:
      return blob / xdnnParams['scaleB']

    blob = np.ascontiguousarray(inputs[0], dtype=np.float32)
#    print "blobBeforeCaffe",blob.shape
#    blobBeforeCaffe=np.ascontiguousarray(np.transpose(blob,(0,3,1,2)), dtype=np.float32)
#    blobBeforeCaffe=blobBeforeCaffe.flatten()
#    blobBeforeCaffe=blobBeforeCaffe.tolist()
#    open("xdlfEmuConv233ReduceQuantizedOutputs.txt","w").close()
#    with open("xdlfEmuConv233ReduceQuantizedOutputs.txt","w") as fIter:
##      fIter.write("max min avg stddev "+str(max(blobBeforeCaffe))+" "+str(min(blobBeforeCaffe))+" "+str(float(sum(blobBeforeCaffe))/float(len(blobBeforeCaffe)))+" "+str(np.array(blobBeforeCaffe).std())+"\n")
#      for i in range(len(blobBeforeCaffe)):
#        fIter.write(str(i)+" "+str(int(blobBeforeCaffe[i]))+'\n')
    qp = xdnnParams['quantDB'][self.quantize_key]
    xdnnParams['api'].unquantizeTensor(\
      qp['th_layer_out'], qp['bw_params'], blob)

#    blobAfterCaffe=np.ascontiguousarray(np.transpose(blob,(0,3,1,2)), dtype=np.float32)
#    blobAfterCaffe=blobAfterCaffe.flatten()
#    fName=self.output.strip
#    np.save('xdlfEmu'+self.output.replace('/','_')+'UnquantizedOutputs.npy', blobAfterCaffe)
#    a=np.load('xdlfEmu'+self.output.replace('/','_')+'UnquantizedOutputs.npy')
#    b=np.load('xdlf'+self.output.replace('/','_')+'FpgaOutputxdnnv3.npy')
#    print "MNDBG np all close",self.output,((a - b) ** 2).mean(axis=None)
#
#
#
#    blobAfterCaffe=blobAfterCaffe.tolist()
#    open("xdlfEmuConv233ReduceUnQuantizedOutputs.txt","w").close()
#    with open("xdlfEmuConv233ReduceUnQuantizedOutputs.txt","w") as fIter:
#      for i in range(len(blobAfterCaffe)):
#        fIter.write(str(i)+" "+str(blobAfterCaffe[i])+'\n')
    elapsedTime = timeit.default_timer() - startTime
    #print "[time] unquantize_layer: %.2fms" % (elapsedTime*1000)

    return blob
