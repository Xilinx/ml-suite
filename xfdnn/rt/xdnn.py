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
#!/usr/bin/python
from ctypes import *
import os
import timeit
import numpy as np

# Models a set of PEs, their buffers and ScriptExecutor.
class XDNNExecData:
  def __init__(self, peMask, peIdxList):
    self._peMask = peMask
    self._peIdxList = peIdxList
    self._networkId = None
    self._networkTimestamp = None
    self._executor = None
    self._cpuInputs = None
    self._cpuInputPtrs = None
    self._fpgaInputs = None
    self._fpgaInputPtrs = None
    self._fpgaInputPtrsOrig = None
    self._fpgaInputHandles = None
    self._numBatches = None
    self._numImgPerBatch = None
    self._imgSize = None
    self._pendingJob = None

class XDNNManager:
  def __init__(self, libFile=None):
    if not libFile and "XDNN_LIB_PATH" in os.environ:
      libPath = os.environ["XDNN_LIB_PATH"]
      if os.path.isfile(libPath+"/libxblas.so"):
        libFile = libPath+"/libxblas.so"
      elif os.path.isfile(libPath+"/libxfdnn.so"):
        libFile = libPath+"/libxfdnn.so"
      
    if not libFile:
      raise AssertionError("XDNN library .so file not found")

    self._handles = None
    self._execData = {}

    self._lib = cdll.LoadLibrary(libFile)
    self._lib.xMalloc.restype = c_void_p 
    self._lib.xMemcpyHost2Device.argtypes \
      = [c_void_p, c_void_p, c_void_p, c_int]
    self._lib.XDNNFillWeightsBiasQuantBlob.argtypes \
      = [POINTER(c_short), c_int, 
         c_char_p, c_char_p,
         POINTER(c_float), c_uint, c_float,
         POINTER(c_float), c_uint, c_float,
         c_ushort, c_ushort, c_uint, c_uint]
    self._lib.XDNNMakeWeightsBiasQuantBlob.restype = POINTER(c_short)
    self._lib.xblasLoadA.argtypes \
      = [c_void_p, c_int, 
         c_void_p, c_void_p, c_int]
    self._lib.XDNNPrepareInput.argtypes \
      = [c_char_p,
         POINTER(POINTER(c_float)), 
         POINTER(POINTER(c_short)),
         c_int, c_int, c_char_p,
         c_float]
    self._lib.XDNNPrepareInput.restype = c_int

    self._lib.XDNNMakeScriptExecutor.argtypes \
      = [POINTER(c_void_p), c_int,
         POINTER(c_short), c_char_p, c_char_p, c_float,
         c_int, c_int, c_int]
    self._lib.XDNNMakeScriptExecutor.restype = c_void_p
    self._lib.XDNNUpdateScriptExecutor.argtypes \
      = [c_void_p, POINTER(c_short), c_char_p]
    self._lib.XDNNExecute.argtypes \
      = [c_void_p,
         POINTER(POINTER(c_short)), c_void_p,
         c_int, c_bool]

    self._lib.XDNNQuantizeTensor.argtypes = [c_float, c_int,
      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]
    self._lib.XDNNUnQuantizeTensor.argtypes = [c_float, c_int,
      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]
    self._lib.XDNNQuantizeInterLayer.argtypes \
      = [c_int, c_int, c_int, c_int, 
         np.ctypeslib.ndpointer(c_longlong, flags="C_CONTIGUOUS"), c_int]
    self._lib.XDNNQuantizeBias.argtypes = [c_float, c_int, c_float]
    self._lib.XDNNQuantizeWeights.argtypes = [c_float, c_int, 
      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]

    self._lib.computeFC.argtypes \
      = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
         c_int, c_int, c_int, 
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")]
    self._lib.XDNNWaitForResults.argtypes = [c_void_p]

    self._lib.XDNNReadWeightsFile.argtypes \
      = [c_char_p, POINTER(POINTER(c_char)), 
         POINTER(POINTER(c_int)), POINTER(POINTER(c_int)),
         POINTER(POINTER(c_int)), POINTER(POINTER(c_int)), 
         POINTER(POINTER(c_int)), POINTER(POINTER(c_float))]

  def createHandle(self, xclbin, kernel, numHandles):
    self._handles = []

    ret = 0
    for i in range(numHandles):
      self._handles.append(c_void_p())
      ret |= self._lib.xblasCreate(pointer(self._handles[i]),
        c_char_p(xclbin), c_char_p(kernel), i)

    return ret

  def closeHandle(self): 
    if not self._handles:
      return

    for h in self._handles:
      self._lib.xblasDestroy(h)

  def getMask(self, peList):
    if not isinstance(peList, list): peList = [peList]

    peMask = 0
    for peId in peList:
      if peId == -1: return 0
      peMask = peMask | (1 << peId)
    return peMask

  # PE is usually a single int referring to a PE. 
  # It can also be a list of ints for advanced usage. 
  # Verifies that all PEs are supported by the FPGA handle
  # and raises AssertionError otherwise. 
  # Verifies that the PE is disjoint with all current executors.
  # returns a unique "execData" for the PE/PE-list
  def getOrCreateExecData(self, PE):
    if not isinstance (PE, list): PE = [PE]
    peMask = self.getMask(PE)

    if peMask in self._execData:
      return self._execData[peMask]

    # check if any element of PE is already in a different executor. 
    # Mask=0 => all PEs used
    for key in self._execData.keys():
      if (key & peMask) or (peMask == 0) or (key == 0):
        if self._execData[key]._pendingJob:
          raise AssertionError("PE is non-disjoint and is already in use")
        else:
          # delete old assignment
          del self._execData[key]

    # save new PE assignment
    self._execData[peMask] = XDNNExecData(peMask, PE) 
    return self._execData[peMask]

  def initScriptExecutor(self, weightsBlob, netFile, cfgFile, scale,
    numBatches, numImgPerBatch, execData):

    netFileTimestamp = os.stat(netFile).st_mtime

    if not execData._executor:
      numHandles = len(self._handles)
      handlePtrs = (c_void_p*numHandles)()
      for i,h in enumerate(self._handles):
        handlePtrs[i] = h

      execData._executor = self._lib.XDNNMakeScriptExecutor(\
        handlePtrs, numHandles, weightsBlob, netFile, cfgFile, 
        scale, numBatches, numImgPerBatch, execData._peMask)
    elif execData._networkId != netFile \
      or execData._networkTimestamp != netFileTimestamp:
      self._lib.XDNNUpdateScriptExecutor(\
        execData._executor, weightsBlob, netFile)
    else:
      # reuse existing executor
      pass

    execData._networkId = netFile
    execData._networkTimestamp = netFileTimestamp
    execData._numBatches = numBatches
    execData._numImgPerBatch = numImgPerBatch

  def makeFPGAShortArray(self, n):
    cp = np.ascontiguousarray(np.zeros(n), dtype=np.int16)
    size = cp.nbytes
    fp = None
    if self._handles:
      for h in self._handles:
        fp = self._lib.xMalloc(h, size, True)
        self._lib.xMemcpyHost2Device(h, 
          cp.ctypes.data_as(POINTER(c_short)), fp, size)
    return (cp, fp)

  def makeFPGAFloatArray(self, n):
    cp = np.ascontiguousarray(np.zeros(n), dtype=np.float32)
    size = cp.nbytes
    fp = None
    if self._handles:
      for h in self._handles:
        fp = self._lib.xMalloc(h, size, True)
        self._lib.xMemcpyHost2Device(h, 
          cp.ctypes.data_as(POINTER(c_float)), fp, size)
    return (cp, fp)

  def markDirty(self, peMask):
    for i in range(len(self._execData[peMask]._fpgaInputPtrs)):
      cp = self._execData[peMask]._fpgaInputPtrs[i]
      fp = self._execData[peMask]._fpgaInputHandles[i]
      size = self._execData[peMask]._fpgaInputs[i].nbytes
      if self._handles:
        for h in self._handles:
          self._lib.xMemcpyHost2Device(h, cp, fp, size)

  def fillWeightsBiasQuantBlob(self, blob, offset, 
    cfgFile, weights, scaleWeight, bias, scaleBias, kw, kh, inch, outch, layerName = ""):
    cWeights = (c_float*len(weights))()
    for i in range(len(weights)):
      cWeights[i] = weights[i]
    cBias = (c_float*len(bias))()
    for i in range(len(bias)):
      cBias[i] = bias[i]

    return self._lib.XDNNFillWeightsBiasQuantBlob(\
      blob, offset, c_char_p(layerName),
      c_char_p(cfgFile), 
      cWeights, len(cWeights), scaleWeight,
      cBias, len(cBias), scaleBias,
      c_ushort(kw), c_ushort(kh), inch, outch)

  def loadBlobToDdr(self, blob, size, PE):
    execData = self.getOrCreateExecData(PE)
    
    numBytes = size * 2
    for h in self._handles:
      fp = self._lib.xMalloc(h, numBytes, True)
      self._lib.xMemcpyHost2Device(h, blob, fp, numBytes)
    
    for peIdx in execData._peIdxList: 
      for h in self._handles:
        self._lib.xblasLoadA(h, size, blob, None, peIdx)

  def prepareInputsForFpga(self, inputs, cfgFile, scale, PE=-1, layerName=""):
    startTime = timeit.default_timer()

    execData = self.getOrCreateExecData(PE)
    
    numBatches, imgSize = inputs.shape
    #print "imgSize: %s, numBatch: %s" % (imgSize, numBatches)

    inputPtrsNeedInit = (type(execData._cpuInputs) == type(None) \
      or execData._cpuInputs.shape[0] != numBatches \
      or type(execData._fpgaInputs) == type(None) \
      or len(execData._fpgaInputs) != numBatches \
      or execData._imgSize != imgSize)

    if inputPtrsNeedInit:
      # prepare src float array
      execData._imgSize = imgSize
      execData._cpuInputs = np.ascontiguousarray(\
        np.zeros(numBatches*imgSize).reshape(numBatches, imgSize),
        dtype=np.float32)

      # prepare array of ptrs to each input image
      execData._cpuInputPtrs = (POINTER(c_float)*numBatches)()
      for i in range(numBatches):
        execData._cpuInputPtrs[i] \
          = execData._cpuInputs[i].ctypes.data_as(POINTER(c_float))

      # prepare tgt short array
      execData._fpgaInputPtrs = (POINTER(c_short)*numBatches)()
      execData._fpgaInputPtrsOrig = execData._fpgaInputPtrs
      execData._fpgaInputs = []
      execData._fpgaInputHandles = []
      for i in range(numBatches):
        (fpgaArr, fpgaHandle) = self.makeFPGAShortArray(imgSize)
        execData._fpgaInputs.append(fpgaArr)
        execData._fpgaInputPtrs[i] \
          = execData._fpgaInputs[i].ctypes.data_as(POINTER(c_short))
        execData._fpgaInputHandles.append(fpgaHandle)
    else:
      execData._fpgaInputPtrs = execData._fpgaInputPtrsOrig
  
    if inputs.dtype == np.float32:
      # grab new input data
      for i in range(numBatches):
        np.copyto(execData._cpuInputs[i], inputs[i])
      
      # prepare data for FPGA
      actualNumFpgaInputs = self._lib.XDNNPrepareInput(\
        layerName, execData._cpuInputPtrs, 
        execData._fpgaInputPtrs, 
        numBatches, imgSize, cfgFile,
        scale)

      if actualNumFpgaInputs < len(execData._fpgaInputPtrs):
        # quantized/packed
        truncatedArr = (POINTER(c_short)*actualNumFpgaInputs)()
        for i in range(actualNumFpgaInputs):
          truncatedArr[i] = execData._fpgaInputPtrs[i]
        execData._fpgaInputPtrs = truncatedArr
    else:
      # already prepared, just populate fields 
      for i in range(numBatches):
        np.copyto(execData._fpgaInputs[i], inputs[i])

    # tell FPGA there's new data
    self.markDirty(execData._peMask)
    elapsedTime = timeit.default_timer() - startTime
    #print "PrepareInputsForFpga elapsed (%f ms):" % (elapsedTime * 1000)
      
    return execData._fpgaInputPtrs

  def execute(self, netFile, weightsBlob, inputs, output,
    numBatches, numImgPerBatch, cfgFile, scale, PE, blocking=True):

    outputPtr = None
    if isinstance(output, np.ndarray):
      outputPtr = output.ctypes.data_as(c_void_p)
    else:
      outputPtr = output

    execData = self.getOrCreateExecData(PE)

    # get/create script executor and reconfig it if necessary
    self.initScriptExecutor(weightsBlob, netFile, cfgFile, 
      scale, numBatches, numImgPerBatch, execData)

    numFpgaBatches = len(inputs)

    if not blocking:
      execData._pendingJob = True

    return self._lib.XDNNExecute(execData._executor, 
      inputs, outputPtr, numFpgaBatches, blocking)

  def exec_async (self, netFile, weightsBlob, inputs, output, 
    numBatches, numImgPerBatch, cfgFile, scale, PE):
    return self.execute(netFile, weightsBlob, inputs, output,
      numBatches, numImgPerBatch, cfgFile, scale, PE, blocking=False)

  def get_result(self, PE):
    peMask = self.getMask(PE);

    if peMask not in self._execData:
      print 'peMask '+str(peMask)+' keys '+str (self._execData.keys())
      raise AssertionError("ExecData not found for PE " + str(PE))
    #print "waitForResults peMask %d\n" % peMask
    ret = self._lib.XDNNWaitForResults(self._execData[peMask]._executor)
    self._execData[peMask]._pendingJob = False
    return ret

  def readWeightsFile(self, fname):
    layerNamePtr = POINTER(c_char)() 
    kwPtr = POINTER(c_int)()
    khPtr = POINTER(c_int)()
    icPtr = POINTER(c_int)()
    ocPtr = POINTER(c_int)()
    nvPtr = POINTER(c_int)()
    valsPtr = POINTER(c_float)()

    self._lib.XDNNReadWeightsFile(fname, byref(layerNamePtr), 
      byref(kwPtr), byref(khPtr),
      byref(icPtr), byref(ocPtr), byref(nvPtr), byref(valsPtr))

    # convert to Python
    layerName = ""
    i = 0
    while layerNamePtr[i] != '\0':
      layerName += layerNamePtr[i]
      i += 1
    kw = kwPtr[0]
    kh = khPtr[0]
    ic = icPtr[0]
    oc = ocPtr[0]
    nv = nvPtr[0]
    vals = []
    for i in range(nv):
      vals.append(valsPtr[i])

    return (layerName, kw, kh, ic, oc, vals)

  def quantizeInterLayer(self, preShift, scale, postShift, bitWidth, v):
    return self._lib.XDNNQuantizeInterLayer(\
      preShift, scale, postShift, bitWidth, v, v.size)

  def quantizeWeights(self, thresh, bitWidth, v):
    return self._lib.XDNNQuantizeWeights(thresh, bitWidth, v, v.size)

  def quantizeBias(self, threshOut, bitWidth, val):
    return self._lib.XDNNQuantizeBias(threshOut, bitWidth, val)

  def quantizeTensor(self, threshIn, bitWidth, v):
    return self._lib.XDNNQuantizeTensor(threshIn, bitWidth, v, v.size)

  def unquantizeTensor(self, threshOut, bitWidth, v):
    return self._lib.XDNNUnQuantizeTensor(threshOut, bitWidth, v, v.size)

_xdnnManager = None

def createManager ( libFile ):
  global _xdnnManager
  if not _xdnnManager:
    _xdnnManager = XDNNManager(libFile)
  return True
    
def createHandle(xclbin, kernel, libFile, numHandles=1):
  createManager (libFile)
  return _xdnnManager.createHandle(xclbin, kernel, numHandles)

def closeHandle():
  return _xdnnManager.closeHandle()

def makeFPGAFloatArray(n):
  return _xdnnManager.makeFPGAFloatArray(n)

def computeWeightsBiasQuantSize(kWidth, kHeight, inCh, outCh, doQuant):
  return _xdnnManager._lib.XDNNComputeWeightsBiasQuantSize(\
    kWidth, kHeight, inCh, outCh, doQuant)

def makeWeightsBiasQuantBlob(size):
  return _xdnnManager._lib.XDNNMakeWeightsBiasQuantBlob(size)

def fillWeightsBiasQuantBlob(blob, offset, cfgFile, 
  weights, scaleWeight, bias, scaleBias, 
  kw, kh, inch, outch, layerName = ""):
  return _xdnnManager.fillWeightsBiasQuantBlob(\
    blob, offset, cfgFile, 
    weights, scaleWeight, bias, scaleBias, kw, kh, inch, outch, layerName)

def loadBlobToDdr(blob, size, PE=-1):
  return _xdnnManager.loadBlobToDdr(blob, size, PE)

# this is typically the first method that is called.
def prepareInputsForFpga(inputs, cfgFile, scale, PE=-1, layerName = ""):
  return _xdnnManager.prepareInputsForFpga( inputs, cfgFile, scale, PE, layerName)

# TODO would like to use None as default. But it matches 0, a valid PE
def execute(netFile, weightsBlob, inputs, output, numBatches, 
  numImgPerBatch, cfgFile, scale, PE=-1):
  return _xdnnManager.execute(netFile, weightsBlob, inputs, output, 
    numBatches, numImgPerBatch, cfgFile, scale, PE)

def exec_async (netFile, weightsBlob, inputs, output, numBatches, 
  numImgPerBatch, cfgFile, scale, PE=-1):
  return _xdnnManager.exec_async (netFile, weightsBlob, inputs, output, 
    numBatches, numImgPerBatch, cfgFile, scale, PE)

def get_result(PE=-1):
  return _xdnnManager.get_result(PE)

def softmax(data):
  import math
  maxVal = data.max()
  for x in np.nditer(data, op_flags=['readwrite']):
    x[...] = math.exp(x - maxVal)

  totalVal = np.sum(data)
  for x in np.nditer(data, op_flags=['readwrite']):
    x[...] = x / totalVal

  return data    

def computeSoftmax(data, num):
  outSize = len(data) / num

  i = 0
  for n in range(num):
    data[i:i+outSize] = softmax(data[i:i+outSize])
    i += outSize

  return data
  
def computeFC(weight, bias, data, M, N, K, useBlas):
  M = int(M)
  N = int(N)
  K = int(K)
  if len(weight) != K*N:
    raise Exception('FC weight dim mismatch')
  if len(data) != M*K:
    raise Exception('FC input dim mismatch')
  if len(bias) != N:
    raise Exception('FC bias dim mismatch')

  if useBlas:
    # Use BLAS
    size = M * N
    output = np.ascontiguousarray(np.zeros(size), dtype=np.float32)
    _xdnnManager._lib.computeFC(weight, bias, data, M, N, K, output)
    return output

  # make 2-D arrays
  inMat = data.reshape(M, K)
  # Note: Caffe's FC weights are already transposed
  weightMat = weight.reshape(N, K).transpose()

  output = np.dot(inMat, weightMat)

  # add bias
  for i in range(output.shape[0]):
    for j in range(output.shape[1]):
      output[i][j] += bias[j]

  return output

def quantizeInputs(layerName, 
  inputs, inputsPtr, fpgaInputsPtr, cfgFile, scale):
  numBatches, imgSize = inputs.shape

  if type(inputsPtr) == type(None):
    # first time -- allocate ctypes mem
    inputsPtr = (POINTER(c_float)*numBatches)()
    fpgaInputsPtr = (POINTER(c_short)*numBatches)()
    for i in range(numBatches):
      inputsPtr[i] = (c_float*imgSize)()
      fpgaInputsPtr[i] = (c_short*imgSize)()

  for i in range(numBatches):
    inputsPtr[i] = inputs[i].ctypes.data_as(POINTER(c_float))
  
  numFpgaInputs = _xdnnManager._lib.XDNNPrepareInput(\
    layerName, inputsPtr, fpgaInputsPtr,
    numBatches, imgSize, cfgFile, scale)

  # make numpy arrays from ctypes arrays
  blobs = []
  for i in range(numFpgaInputs):
    arr = np.frombuffer(\
      (c_short*imgSize).from_address(\
          addressof(fpgaInputsPtr[i].contents)), np.int16)
    blobs.append(arr)

  result = np.vstack(blobs)
  return result

def readWeightsFile(fname):
  return _xdnnManager.readWeightsFile(fname)
