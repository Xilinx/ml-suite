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
import matop_layer
import pool_layer

class bunch_fpga_layer(conv_layer.conv_layer):
  def __init__(self, weights = None, stride = [1,1,1,1], 
    activation = None, padding = False, biases = 0, 
    quantize_key="", xdnn_env=None, bunchXdlflayers=None) :
    super(bunch_fpga_layer, self).__init__(weights, stride, activation, padding, biases)
    self.quantize_key = quantize_key
    self.xdnn_env = xdnn_env
    self.bunchXdlfLayers=bunchXdlflayers
  
  def setBunchXdlfLayers(self, bunchXdlflayers):
    self.bunchXdlfLayers=bunchXdlflayers
    for i in self.bunchXdlfLayers:
      print("MNDBG bunch layers",i.output)


  def set_params(self, layer_params, variables, 
    quantize_key="", xdnn_env=None) :
    super(bunch_fpga_layer, self).set_params(layer_params, variables)
    self.quantize_key = quantize_key
    self.xdnn_env = xdnn_env
    return self

  def forward_exec(self,inputs) :
    print("Accelerating on FPGA: %s" % self.output)

    xdnnParams = self.xdnn_env.get_params()
    args = {
      'quantizecfg': xdnnParams['quantize_json'],
      'scaleA':10000,
      'scaleB':30,
      'PE':-1,
      'batch_sz':1,
      'firstfpgalayer': self.quantize_key,
    }

    inp = inputs[0] # assuming input is n, h, w, c
    inp = np.copy(inp)
    print(inp.shape)

    batchSizeIn=inp.shape[0]
    inputHIn=inp.shape[1]
    inputWIn=inp.shape[2]
    inChansIn=inp.shape[3]
    inputsFPGAIn=np.zeros((batchSizeIn, np.prod([inChansIn,inputHIn,inputWIn])), dtype=np.float32)
    for i in range(batchSizeIn):
      caffeFormatInputsIn = np.ascontiguousarray(\
             np.transpose(inp[i], (2,0,1)), dtype=np.float32)#CHW
      RawInputsIn=caffeFormatInputsIn.flatten()
      RawInputsIn.tolist()
      for i in range(len(RawInputsIn)):
        RawInputsIn[i] = 1.0
    
    inp=RawInputsIn
    inp=np.asanyarray(inp,dtype=np.float32)
    inp=inp.reshape(batchSizeIn, inChansIn, inputHIn, inputWIn)
    inp=np.ascontiguousarray(np.transpose(inp,(0,2,3,1)),dtype=np.float32)

    inp = np.ascontiguousarray(inp, dtype=np.float32)
    inp=inputs[0] 
    #Tensorflow Weights Format is HWIcOc
#    FPGAFormatWeights,kernW,kernH,inChans,outChans=self.getFPGAFormatWeightsKernVars(args, self.bunchXdlfLayers,xdnnParams['isXdnnv3'])
    if xdnnParams['isXdnnv3']=="True":
      compilerFile = "jul23rdcompilerrepl.cmds"#"jul23rdcompilernorepl.cmds"
    else:
      compilerFile = "xdlfBunchv2.cmd"
    
    if xdnnParams['isXdnnv3']=="True":
      allConvLayerNames, allConvLayerNamesParams = self.getParamsFromCompilerFile(compilerFile)
      print(allConvLayerNames, "MNDBG convparams")
      allLayerNames, allLayersWeightsBiasQuantizeKey, size=self.getFPGAFormatWeightsKernVarsLatestRepl(args, self.bunchXdlfLayers,xdnnParams['isXdnnv3'], allConvLayerNamesParams)
    else:
      allLayerNames, allLayersWeightsBiasQuantizeKey, size=self.getFPGAFormatWeightsKernVars(args, self.bunchXdlfLayers,xdnnParams['isXdnnv3'])
#    print len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'])
    print("MNDBG alllayernames",allLayerNames)
    if xdnnParams['isXdnnv3']=="True":
      (weightsBlob, weightsFpgaHandle) = xdnn_io.XDLFBunchloadWeightsBiasQuantLatestRepl(args, allLayerNames, allLayersWeightsBiasQuantizeKey, allConvLayerNamesParams, size, xdnnParams['isXdnnv3'])
    else:
      (weightsBlob, weightsFpgaHandle) = xdnn_io.XDLFBunchloadWeightsBiasQuant(args, allLayerNames, allLayersWeightsBiasQuantizeKey, size, xdnnParams['isXdnnv3'])
    inputsReq,batchSize,inputH,inputW,inChans=self.XDLFPrepareInputs(inp)
    (fpgaInputs) = xdnn_io.XDLFprepareRawInputs(args,inputsReq,batchSize,inputH,inputW,inp.shape[3])
#    print len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'])
#    self.dumpToFile("xdlfBunchWeightsconv2rFloat.txt",allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'],len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias']))
#    self.dumpToFile("xdlfBunchWeightsconv2Float.txt",allLayersWeightsBiasQuantizeKey['conv2/3x3']['Bias'],len(allLayersWeightsBiasQuantizeKey['conv2/3x3']['Bias']))


    self.dumpToFile("xdlfBunchInpFloat.txt",inputsReq[0],len(inputsReq[0]))
    if batchSize != 1:
      raise NotImplementedError("NOT YET IMPLEMENTED FOR BATCHSIZE>1!!!!!!!!!!!")

    fpgaOutSize = batchSize*int(self.shape[1])*int(self.shape[2])*int(self.shape[3])
    fpgaOutput = xdnn_io.prepareOutput(fpgaOutSize, batchSize)

    #self.makeCompilerFile(\
    #  kernW, kernH, inp.shape, self.shape, xdnnParams['isXdnnv3'])
    print("bunch fpga execution: ") 
    xdnn.execute(compilerFile,weightsBlob,fpgaInputs,fpgaOutput,
              batchSize,args['quantizecfg'],args['scaleB'])
#    fpgaNpySave=np.asanyarray(fpgaOutput,dtype=np.float32)
#
#    np.save('xdlf'+self.output.replace('/','_')+'FpgaOutputxdnnv3jul14th.npy', fpgaNpySave)
#    a=np.load('xdlfEmupool5_7x7_s1UnquantizedOutputsjul14th.npy')
#    b=np.load('xdlf'+self.output.replace('/','_')+'FpgaOutputxdnnv3jul14th.npy')
#    print "MNDBG np mean square error",self.output,((a - b) ** 2).mean(axis=None)
#   self.debugPrintNonzeroFPGAOutputs(fpgaOutput, 100)
    print(batchSize, self.shape[3])
    self.dumpToFile("xdlfBunchfpgaOutputjul14th.txt",fpgaOutput,len(fpgaOutput))
    conv_out=self.getTFFormatOut(fpgaOutput,batchSize,self.shape[3])
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

  def TFlayerName2QuantizeKey(self,name):
    origName = name
    try:
      name = name.split("/", 1)[0]
      underscores = [i for i, ltr in enumerate(name) if ltr == '_']
      name_list = list(name)
      if len(underscores) <= 2:
        if "inception" in name:
          name_list[underscores[1]] = '/'
        else:
          name_list[underscores[0]] = '/'
      elif len(underscores) > 2:
        name_list[underscores[1]] = '/'
      name = ''.join(name_list)
    except:
      name = origName

    return name

  def getParamsFromCompilerFile(self, compilerFileName):
    with open(compilerFileName) as compilerReadStream:
      compilerContent = compilerReadStream.readlines()
    compilerContent = [x.strip().split(" ") for x in compilerContent]
    allLayersWeightsBiasQuantizeKey={}
    layerWeightsBiasQuantizeKey={}
    allLayerNames=[]
    print(compilerContent)
    print(len(compilerContent))
    for i in range(len(compilerContent)):
      print(i, compilerFileName)
      if compilerContent[i][1] == "XNConv":
        print(compilerContent[i][2], "MNDBG insideparams")
        layerWeightsBiasQuantizeKey={}
        if compilerContent[i][2] not in allLayerNames:
          allLayerNames.append(compilerContent[i][2])
        layerWeightsBiasQuantizeKey['kernW']=compilerContent[i][3]
        layerWeightsBiasQuantizeKey['kernH']=compilerContent[i][4]
        layerWeightsBiasQuantizeKey['inChans']=compilerContent[i][19]
        layerWeightsBiasQuantizeKey['outChans']=compilerContent[i][23]
        layerWeightsBiasQuantizeKey['srcFullSectNum']=compilerContent[i][25]
        layerWeightsBiasQuantizeKey['srcReplSectNum']=compilerContent[i][26]
        layerWeightsBiasQuantizeKey['srcReplUnitNum']=compilerContent[i][27]
        layerWeightsBiasQuantizeKey['srcReplUnitWidth']=compilerContent[i][28]
        layerWeightsBiasQuantizeKey['convHalfRateMode']=compilerContent[i][47]

        allLayersWeightsBiasQuantizeKey[compilerContent[i][2]]=layerWeightsBiasQuantizeKey
    return allLayerNames, allLayersWeightsBiasQuantizeKey

  def getFPGAFormatWeightsKernVarsLatestRepl(self, args, bunchLayers, isxdnnv3, convLayersParams):
    allLayersWeightsBiasQuantizeKey={}
    layerWeightsBiasQuantizeKey={}
    allLayerNames=[]
    for i in bunchLayers:
      print(i.output)
      if isinstance(i, conv_layer.conv_layer):
        layerWeightsBiasQuantizeKey={}
        layerName=self.TFlayerName2QuantizeKey(i.output)
        if layerName not in allLayerNames:
          allLayerNames.append(layerName)
        caffeFormatWeights = np.ascontiguousarray(\
          np.transpose(i.filter_weights, (3,2,0,1)), dtype=np.float32)#OcIcHW
        kernW=caffeFormatWeights.shape[3]
        kernH=caffeFormatWeights.shape[2]
        inChans=caffeFormatWeights.shape[1]
        outChans=caffeFormatWeights.shape[0]
        caffeFormatWeights=caffeFormatWeights.flatten()
        caffeFormatWeights=caffeFormatWeights.tolist()
        bias=[0 for v in range(outChans)]
        layerWeightsBiasQuantizeKey['weights']=caffeFormatWeights
        layerWeightsBiasQuantizeKey['kernW']=kernW
        layerWeightsBiasQuantizeKey['kernH']=kernH
        layerWeightsBiasQuantizeKey['inChans']=inChans
        layerWeightsBiasQuantizeKey['outChans']=outChans
        layerWeightsBiasQuantizeKey['Bias']=bias
        allLayersWeightsBiasQuantizeKey[layerName]=layerWeightsBiasQuantizeKey
      elif isinstance(i, matop_layer.matop_layer) \
        and i.optype == "BiasAdd":
        caffeFormatBias=i.Bias.flatten()
        caffeFormatBias=caffeFormatBias.tolist()
#        for j in range(len(caffeFormatBias)):
#          caffeFormatBias[j]=0.0
        layerName=self.TFlayerName2QuantizeKey(i.output)
        if layerName not in list(allLayersWeightsBiasQuantizeKey.keys()):
          raise NotImplementedError("NOT YET IMPLEMENTED FOR ONLY BIAS LAYER!")
        else:
          allLayersWeightsBiasQuantizeKey[layerName]['Bias']=caffeFormatBias  
          self.dumpToFile("xdlfBunchBias"+str(layerName).replace('/','_')+"Float.txt",allLayersWeightsBiasQuantizeKey[layerName]['Bias'],len(allLayersWeightsBiasQuantizeKey[layerName]['Bias']))
#    print len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'])
      elif isinstance(i, pool_layer.pool_layer):
        layerName=self.TFlayerName2QuantizeKey(i.output)

    size=0
    for i in allLayerNames:
      size+= xdnn_io.XDLFBunchComputeSizeLatestRepl(args, allLayersWeightsBiasQuantizeKey[i]['outChans'],allLayersWeightsBiasQuantizeKey[i]['inChans'],allLayersWeightsBiasQuantizeKey[i]['kernH'],allLayersWeightsBiasQuantizeKey[i]['kernW'],  convLayersParams[i]['srcFullSectNum'], convLayersParams[i]['srcReplSectNum'], convLayersParams[i]['srcReplUnitNum'], isxdnnv3)
    
#    print len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'])
    print("MNDBG in func") 
    return allLayerNames, allLayersWeightsBiasQuantizeKey, size


  def getFPGAFormatWeightsKernVars(self, args, bunchLayers, isxdnnv3):
    allLayersWeightsBiasQuantizeKey={}
    layerWeightsBiasQuantizeKey={}
    allLayerNames=[]
    for i in bunchLayers:
      print(i.output)
      if isinstance(i, conv_layer.conv_layer):
        layerWeightsBiasQuantizeKey={}
        layerName=self.TFlayerName2QuantizeKey(i.output)
        if layerName not in allLayerNames:
          allLayerNames.append(layerName)
        caffeFormatWeights = np.ascontiguousarray(\
          np.transpose(i.filter_weights, (3,2,0,1)), dtype=np.float32)#OcIcHW
        kernW=caffeFormatWeights.shape[3]
        kernH=caffeFormatWeights.shape[2]
        inChans=caffeFormatWeights.shape[1]
        outChans=caffeFormatWeights.shape[0]
        caffeFormatWeights=caffeFormatWeights.flatten()
        caffeFormatWeights=caffeFormatWeights.tolist()
        bias=[0 for v in range(outChans)]
        layerWeightsBiasQuantizeKey['weights']=caffeFormatWeights
        layerWeightsBiasQuantizeKey['kernW']=kernW
        layerWeightsBiasQuantizeKey['kernH']=kernH
        layerWeightsBiasQuantizeKey['inChans']=inChans
        layerWeightsBiasQuantizeKey['outChans']=outChans
        layerWeightsBiasQuantizeKey['Bias']=bias
        allLayersWeightsBiasQuantizeKey[layerName]=layerWeightsBiasQuantizeKey
      elif isinstance(i, matop_layer.matop_layer) \
        and i.optype == "BiasAdd":
        caffeFormatBias=i.Bias.flatten()
        caffeFormatBias=caffeFormatBias.tolist()
#        for j in range(len(caffeFormatBias)):
#          caffeFormatBias[j]=0.0
        layerName=self.TFlayerName2QuantizeKey(i.output)
        if layerName not in list(allLayersWeightsBiasQuantizeKey.keys()):
          raise NotImplementedError("NOT YET IMPLEMENTED FOR ONLY BIAS LAYER!")
        else:
          allLayersWeightsBiasQuantizeKey[layerName]['Bias']=caffeFormatBias  
          self.dumpToFile("xdlfBunchBias"+str(layerName).replace('/','_')+"Float.txt",allLayersWeightsBiasQuantizeKey[layerName]['Bias'],len(allLayersWeightsBiasQuantizeKey[layerName]['Bias']))
#    print len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'])
      elif isinstance(i, pool_layer.pool_layer):
        layerName=self.TFlayerName2QuantizeKey(i.output)

    size=0
    for i in allLayerNames:
      size+= xdnn_io.XDLFBunchComputeSize(args, allLayersWeightsBiasQuantizeKey[i]['outChans'],allLayersWeightsBiasQuantizeKey[i]['inChans'],allLayersWeightsBiasQuantizeKey[i]['kernH'],allLayersWeightsBiasQuantizeKey[i]['kernW'], isxdnnv3)
    
#    print len(allLayersWeightsBiasQuantizeKey['conv2/3x3_reduce']['Bias'])
    print("MNDBG in func") 
    return allLayerNames, allLayersWeightsBiasQuantizeKey, size
  
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
