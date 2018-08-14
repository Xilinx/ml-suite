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

import collections
import numpy as np
import math
import os
import conv_layer
import timeit
import xdnn
import xdnn_io
import sys

class conv_fpga_layer(conv_layer.conv_layer):
  def __init__(self, weights = None, stride = [1,1,1,1], 
    activation = None, padding = False, biases = 0, 
    quantize_key="", xdnn_env=None) :
    super(conv_fpga_layer, self).__init__(weights, stride, activation, padding, biases)
    self.quantize_key = quantize_key
    self.xdnn_env = xdnn_env

  def set_params(self, layer_params, variables, 
    quantize_key="", xdnn_env=None) :
    super(conv_fpga_layer, self).set_params(layer_params, variables)
    self.quantize_key = quantize_key
    self.xdnn_env = xdnn_env
    return self

  def forward_exec(self,inputs) :
    print("Accelerating on FPGA: %s" % self.output)

    inp = inputs[0] # assuming input is n, h, w, c

    xdnnParams = self.xdnn_env.get_params()
    quantizeCfg = ""
    if "quantize_json" in xdnnParams:
      quantizeCfg = xdnnParams["quantize_json"]

    args = {
      'quantizecfg': quantizeCfg,
      'scaleA': xdnnParams['scaleA'],
      'scaleB': xdnnParams['scaleB'],
      'PE':-1,
      'batch_sz':1,
      'firstfpgalayer': self.quantize_key,
    }

    #Tensorflow Weights Format is HWIcOc
    FPGAFormatWeights,kernW,kernH,inChans,outChans=self.getFPGAFormatWeightsKernVars(self.filter_weights)

    if outChans != self.shape[3]:
      raise RuntimeError("output channel mismatch %s %d != %d" \
        % (self.output, outChans, self.shape[3]))

    (weightsBlob, weightsFpgaHandle) = xdnn_io.XDLFloadWeights(args,FPGAFormatWeights,outChans,inChans,kernH,kernW, self.quantize_key, xdnnParams['isXdnnv3'])
    inputsReq,batchSize,inputH,inputW,inChans=self.XDLFPrepareInputs(inp)
    (fpgaInputs) = xdnn_io.XDLFprepareRawInputs(args,inputsReq,batchSize,inputH,inputW,inp.shape[3])

    if batchSize != 1:
      raise NotImplementedError("NOT YET IMPLEMENTED FOR BATCHSIZE>1!!!!!!!!!!!")

    fpgaOutSize = batchSize*int(self.shape[1])*int(self.shape[2])*int(self.shape[3])
    fpgaOutput = xdnn_io.prepareOutput(fpgaOutSize, batchSize)

    compilerFile = self.makeCompilerFile(\
      kernW, kernH, inp.shape, self.shape, xdnnParams['isXdnnv3'])
    
    xdnn.execute(compilerFile,weightsBlob,fpgaInputs,fpgaOutput,
              batchSize,args['quantizecfg'],args['scaleB'])
#    fpgaNpySave=np.asanyarray(fpgaOutput,dtype=np.float32)

#    np.save('xdlf'+self.output.replace('/','_')+'FpgaOutputxdnnv3.npy', fpgaNpySave)
#    a=np.load('xdlfEmu'+self.output.replace('/','_')+'UnquantizedOutputs.npy')
#    b=np.load('xdlf'+self.output.replace('/','_')+'FpgaOutputxdnnv3.npy')
#    print "MNDBG np mean square error",self.output,((a - b) ** 2).mean(axis=None)
#   self.debugPrintNonzeroFPGAOutputs(fpgaOutput, 100)
    conv_out=self.getTFFormatOut(fpgaOutput,batchSize,outChans)
    return conv_out


  def dumpToFile(self,fname,listToDump,listSize):
    open(fname,'w').close()
    with open(fname,'w') as fIter:
      for i in range(len(listToDump)):
        fIter.write(str(i)+' '+str(listToDump[i])+'\n')


  def makeCompilerFile(self, kernW, kernH, inShape, outShape, isxdnnv3=False):
    compilerFile = "compilerOneInstruction.cmd"
    
    padH = padW = 0
    if self.padding:
      padH = (self.filter_weights.shape[0]-1)/2
      padW = (self.filter_weights.shape[1]-1)/2

    f = open(compilerFile, "w")
    if isxdnnv3=="True":
      f.write("1 XNConv %s %d %d "
        "%d %d %d %d 1 1 16 26 2 "
        "0 0 0x0 %d %d %d "
        "%s %d %d %d 0" %
        (self.quantize_key, kernW, kernH, 
        self.conv_stride[2], self.conv_stride[1], padW, padH, 
        inShape[2], inShape[1], inShape[3],hex(int(inShape[2]*inShape[1]*math.ceil(float(inShape[3])/float(96)))),
       outShape[2], outShape[1], outShape[3]))
      f.close()
      return compilerFile
    else:
      f.write("1 XNConv %s %d %d "
        "%d %d %d %d 1 1 16 26 2 "
        "0 0 0x0 %d %d %d "
        "0x70000 %d %d %d 0" %
        (self.quantize_key, kernW, kernH, 
        self.conv_stride[2], self.conv_stride[1], padW, padH, 
        inShape[2], inShape[1], inShape[3],
       outShape[2], outShape[1], outShape[3]))
      f.close()
      return compilerFile

  def getFPGAFormatWeightsKernVars(self,weights):
    caffeFormatWeights = np.ascontiguousarray(\
      np.transpose(weights, (3,2,0,1)), dtype=np.float32)#OcIcHW
    kernW=caffeFormatWeights.shape[3]
    kernH=caffeFormatWeights.shape[2]
    inChans=caffeFormatWeights.shape[1]
    outChans=caffeFormatWeights.shape[0]
    caffeFormatWeights=caffeFormatWeights.flatten()
    caffeFormatWeights=caffeFormatWeights.tolist()
#    maxi=max(caffeFormatWeights)
#    mini=min(caffeFormatWeights)
#    avg=float(sum(caffeFormatWeights))/float(len(caffeFormatWeights))
#    print "MNDBG COnv233Reduce max min avg stddev weights ",maxi, mini, avg, np.array(caffeFormatWeights).std()
   # for i in range(len(caffeFormatWeights)):
   #   if i%2==0:
   #     caffeFormatWeights[i]=1.0
   #   else:
   #     caffeFormatWeights[i]=-1.0
    return caffeFormatWeights,kernW,kernH,inChans,outChans
  
  def XDLFPrepareInputs(self,inputs):
    batchSize=inputs.shape[0]
    inputH=inputs.shape[1]
    inputW=inputs.shape[2]
    inChans=inputs.shape[3]
    inputsFPGA=np.zeros((batchSize, np.prod([inChans,inputH,inputW])), dtype=np.float32)
    for i in range(batchSize):
      caffeFormatInputs = np.ascontiguousarray(\
             np.transpose(inputs[i], (2,0,1)), dtype=np.float32)#CHW
      RawInputs=caffeFormatInputs.flatten()
      RawInputs.tolist()
#      print "MNDBG COnv233Reduce max min avg stddev inputs",max(RawInputs), min(RawInputs), float(sum(RawInputs))/float(len(RawInputs)), np.array(RawInputs).std()
#      for j in range(len(RawInputs)):
#        if j<49:
#          RawInputs[j]=3.0
#        else:
#          RawInputs[j]=116.0
      inputsFPGA[i]=RawInputs
    return inputsFPGA,batchSize,inputH,inputW,inChans
  
  def debugPrintNonzeroFPGAOutputs(self,fpgaOutput, n=100):
    counter=0
    for i in range(len(fpgaOutput)):
      if fpgaOutput[i]>0:
        if counter<n:
          print(i,fpgaOutput[i])
          counter=counter+1
  
  def getTFFormatOut(self,fpgaOutput,batchSize,outChans):
    outH = self.shape[1]
    outW = self.shape[2]
    outCh = self.shape[3]
    fpgaOutput=np.asanyarray(fpgaOutput,dtype=np.float32)
    fpgaOutput=fpgaOutput.reshape(batchSize, outCh, outH, outW)
    fpgaOutput=np.ascontiguousarray(np.transpose(fpgaOutput,(0,2,3,1)),dtype=np.float32)
    conv_out=fpgaOutput  
    return conv_out
