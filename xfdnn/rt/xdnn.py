#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from ctypes import *
import os
import timeit
import numpy as np

def _makeExecDataKey(peMask, streamId):
  return "%d:%d" % (peMask, streamId)

# Models a set of PEs, their buffers and ScriptExecutor.
class XDNNExecData:
  def __init__(self, peMask, peIdxList, key):
    self._key = key
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
    self._lib.xHostMalloc.argtypes = [c_size_t]
    self._lib.xHostMalloc.restype = c_void_p 
    self._lib.xMalloc.argtypes \
      = [c_void_p , c_size_t, c_bool]
    self._lib.xMalloc.restype = c_void_p 
    self._lib.xMemcpyHost2Device.argtypes \
      = [c_void_p, c_void_p, c_void_p, c_size_t]
    self._lib.xFree.argtypes = [c_void_p, c_void_p, c_bool]
    self._lib.XDNNV3FillWeightsBiasQuantBlob.argtypes \
      = [POINTER(c_short), c_int, 
         c_char_p, c_char_p,
         POINTER(c_float), c_uint, c_float,
         POINTER(c_float), c_uint, c_float,
         c_ushort, c_ushort, c_uint, c_uint,
         c_int, c_int, c_int, c_int, c_bool]
    self._lib.XDNNFillWeightsBiasQuantBlob.argtypes \
      = [POINTER(c_short), c_int, 
         c_char_p, c_char_p,
         POINTER(c_float), c_uint, c_float,
         POINTER(c_float), c_uint, c_float,
         c_ushort, c_ushort, c_uint, c_uint]
    self._lib.XDNNMakeWeightsBiasQuantBlob.restype = POINTER(c_short)
    self._lib.xblasLoadA.argtypes \
      = [c_void_p, c_int, 
         c_void_p, c_char_p, c_void_p, c_int]
    self._lib.XDNNPrepareInput.argtypes \
      = [c_char_p,
         POINTER(POINTER(c_float)), 
         POINTER(POINTER(c_short)),
         c_int, c_int, c_char_p,
         c_float]
    self._lib.XDNNPrepareInput.restype = c_int
    
    self._lib.XDNNPrepareInputFlatArray.argtypes \
      = [c_char_p,
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), 
         np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"),
         c_int, c_int, c_char_p,
         c_float]
    self._lib.XDNNPrepareInputFlatArray.restype = c_int

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
         c_int, c_int, c_bool]
    self._lib.XDNNQuantizeAvgPool.argtypes = [c_float, c_float, c_int, c_int]
    self._lib.XDNNQuantizeTensor.argtypes = [c_float, c_int,
      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]
    self._lib.XDNNUnQuantizeTensor.argtypes = [c_float, c_int,
      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]
    self._lib.XDNNV3QuantizeInterLayer.argtypes \
      = [c_int, c_int, c_int, c_int, 
         np.ctypeslib.ndpointer(c_longlong, flags="C_CONTIGUOUS"), c_int, c_int]
    self._lib.XDNNQuantizeInterLayer.argtypes \
      = [c_int, c_int, c_int, c_int, 
         np.ctypeslib.ndpointer(c_longlong, flags="C_CONTIGUOUS"), c_int]
    self._lib.XDNNQuantizeBias.argtypes = [c_float, c_int, c_float]

    self._lib.XDNNV3QuantizeBias.argtypes = [c_float, c_float, c_int, c_float, c_bool]

    self._lib.XDNNQuantizeWeights.argtypes = [c_float, c_int, 
      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_int]

    self._lib.computeFC.argtypes \
      = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
         c_int, c_int, c_int, 
         np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")]
    self._lib.XDNNWaitForResults.argtypes = [c_void_p, c_int]

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
      ret |= self._lib.xblasCreate(
        pointer(self._handles[i]),
        c_char_p(xclbin),
        c_char_p(kernel),
        i)

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
  def getOrCreateExecData(self, PE, streamId):
    if not isinstance (PE, list): PE = [PE]
    peMask = self.getMask(PE)
    execDataKey = _makeExecDataKey(peMask, streamId)

    if execDataKey in self._execData:
      if self._execData[execDataKey]._pendingJob:
        raise AssertionError("PE already in use")
      return self._execData[execDataKey]

    # save new PE/stream assignment
    self._execData[execDataKey] = XDNNExecData(peMask, PE, execDataKey) 
    return self._execData[execDataKey]

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
    size = n*2 # 2 bytes for short
    memAddr = self._lib.xHostMalloc(size)
    cp = np.frombuffer((c_short*n).from_address(memAddr), np.int16)
    fps = []
    if self._handles:
      for h in self._handles:
        fp = self._lib.xMalloc(h, size, True)
        self._lib.xMemcpyHost2Device(h, 
          cp.ctypes.data_as(POINTER(c_short)), fp, size)
        fps.append(fp)
    return (cp, fps)

  def makeFPGAFloatArray(self, n):
    size = n*4 # 4 bytes for float
    memAddr = self._lib.xHostMalloc(size)
    cp = np.frombuffer((c_float*n).from_address(memAddr), np.float32)
    fps = []
    if self._handles:
      for h in self._handles:
        fp = self._lib.xMalloc(h, size, True)
        self._lib.xMemcpyHost2Device(h, 
          cp.ctypes.data_as(POINTER(c_float)), fp, size)
        fps.append(fp)
    return (cp, fps)

  def markDirty(self, execDataKey):
    for i in range(len(self._execData[execDataKey]._fpgaInputPtrs)):
      cp = self._execData[execDataKey]._fpgaInputPtrs[i]
      fps = self._execData[execDataKey]._fpgaInputHandles[i]
      size = self._execData[execDataKey]._fpgaInputs[i].nbytes
      if self._handles:
        for hi, h in enumerate(self._handles):
          self._lib.xMemcpyHost2Device(h, cp, fps[hi], size)

  def v3fillWeightsBiasQuantBlob(self, blob, offset, 
    cfgFile, weights, scaleWeight, bias, scaleBias, kw, kh, inch, outch, srcFullSectNum, srcReplSectNum, srcReplunitNum, srcReplUnitWidth, convHalfRateMode, layerName = ""):
    cWeights = (c_float*len(weights))()
    for i in range(len(weights)):
      cWeights[i] = weights[i]
    cBias = (c_float*len(bias))()
    for i in range(len(bias)):
      cBias[i] = bias[i]

    return self._lib.XDNNV3FillWeightsBiasQuantBlob(\
      blob, offset, c_char_p(layerName),
      c_char_p(cfgFile), 
      cWeights, len(cWeights), scaleWeight,
      cBias, len(cBias), scaleBias,
      c_ushort(kw), c_ushort(kh), inch, outch, srcFullSectNum, srcReplSectNum, srcReplunitNum, srcReplUnitWidth, convHalfRateMode)

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

  def loadBlobToDdr(self, blob, size, layer2OffsetMap, PE):
    if not isinstance (PE, list): PE = [PE]
    
    numBytes = size * 2
    fps = []
    for h in self._handles:
      fp = self._lib.xMalloc(h, numBytes, True)
      self._lib.xMemcpyHost2Device(h, blob, fp, numBytes)
      fps.append(fp)
    
    for peIdx in PE: 
      for h in self._handles:
        self._lib.xblasLoadA(h, size, blob, layer2OffsetMap, None, peIdx)

    return fps

  def prepareInputsForFpga(self, inputs, cfgFile, scale, 
    PE=-1, layerName="", streamId=0):
    startTime = timeit.default_timer()

    execData = self.getOrCreateExecData(PE, streamId)
    
    numBatches = inputs.shape[0]
    imgSize = np.product( inputs.shape[1:])
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

      # free existing mem (if any)
      if execData._fpgaInputHandles:
        for fps in execData._fpgaInputHandles:
          for hi, h in enumerate(self._handles):
            self._lib.xFree(h, fps[hi], True)
      execData._fpgaInputs = []
      execData._fpgaInputHandles = []

      # make new mem
      for i in range(numBatches):
        (fpgaArr, fpgaHandles) = self.makeFPGAShortArray(imgSize)
        execData._fpgaInputs.append(fpgaArr)
        execData._fpgaInputPtrs[i] \
          = execData._fpgaInputs[i].ctypes.data_as(POINTER(c_short))
        execData._fpgaInputHandles.append(fpgaHandles)
    else:
      execData._fpgaInputPtrs = execData._fpgaInputPtrsOrig
  
    if inputs.dtype == np.float32:
      # grab new input data
      for i in range(numBatches):
        np.copyto(execData._cpuInputs[i], inputs[i].flatten())
      
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
    elif inputs.dtype == np.int16:
       # already prepared, just populate fields 
       for i in range(numBatches):
         np.copyto(execData._fpgaInputs[i], inputs[i].flatten())
    else:
      raise NotImplementedError("Only np.float32 input supported")

    # tell FPGA there's new data
    self.markDirty(execData._key)
      
    return execData._fpgaInputPtrs

  def passThruInputsForFpga(self, ctypesArr, 
    numBatches, imgSize, cfgFile, scale, PE=-1, layerName="", streamId=0):
    execData = self.getOrCreateExecData(PE, streamId)

    if type(execData._fpgaInputs) != type(None):
      self.markDirty(execData._key)
      return execData._fpgaInputPtrs

    execData._fpgaInputPtrs = (POINTER(c_short)*numBatches)()
    execData._fpgaInputs = []
    execData._fpgaInputHandles = []
    imgBytes = imgSize*2 # bytes
    for i in range(numBatches):
      for h in self._handles:
        fp = self._lib.xMalloc(h, imgBytes, True)
        ptr = cast(byref(ctypesArr, i*imgBytes), POINTER(c_short))
        nparr = np.frombuffer(ptr, dtype=np.int16)
        self._lib.xMemcpyHost2Device(h, ptr, fp, imgBytes)

        execData._fpgaInputPtrs[i] = ptr
        execData._fpgaInputHandles.append([fp])
        execData._fpgaInputs.append(nparr)

    self.markDirty(execData._key)
    return execData._fpgaInputPtrs

  def initScript(self, netFile, weightsBlob, 
    numBatches, cfgFile, scale, PE, streamId):
    
    numImgPerBatch = 1 # Legacy Constant
    
    execData = self.getOrCreateExecData(PE, streamId)

    # get/create script executor and reconfig it if necessary
    self.initScriptExecutor(weightsBlob, netFile, cfgFile, 
      scale, numBatches, numImgPerBatch, execData)

  def execute(self, netFile, weightsBlob, inputs, output,
    numBatches, cfgFile, scale, PE, streamId, blocking):

    numImgPerBatch = 1 # Constant, removing from execute API

    outputPtr = None
    if isinstance(output, np.ndarray):
      outputPtr = output.ctypes.data_as(c_void_p)
    else:
      outputPtr = output

    execData = self.getOrCreateExecData(PE, streamId)

    # get/create script executor and reconfig it if necessary
    self.initScriptExecutor(weightsBlob, netFile, cfgFile, 
      scale, numBatches, numImgPerBatch, execData)

    numFpgaBatches = len(inputs)

    if not blocking:
      execData._pendingJob = True

    #startTime = timeit.default_timer()
    result = self._lib.XDNNExecute(execData._executor, 
      inputs, outputPtr, numFpgaBatches, streamId, blocking)
    #endTime = timeit.default_timer()
    #print("ANDBG py.xdnn.execute run " + str((endTime-startTime)*1000))

    return result

  def get_result(self, PE, streamId):
    peMask = self.getMask(PE);
    execDataKey = _makeExecDataKey(peMask, streamId)

    if execDataKey not in self._execData:
      raise AssertionError("ExecData not found for key %s" % execDataKey)

    ret = self._lib.XDNNWaitForResults(self._execData[execDataKey]._executor, streamId)
    self._execData[execDataKey]._pendingJob = False
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

  def quantizev3InterLayer(self, preShift, scale, postShift, bitWidth, v, bias):
    return self._lib.XDNNV3QuantizeInterLayer(\
      preShift, scale, postShift, bitWidth, v, v.size, bias)

  def quantizeInterLayer(self, preShift, scale, postShift, bitWidth, v):
    return self._lib.XDNNQuantizeInterLayer(\
      preShift, scale, postShift, bitWidth, v, v.size)

  def quantizeWeights(self, thresh, bitWidth, v):
    return self._lib.XDNNQuantizeWeights(thresh, bitWidth, v, v.size)

  def quantizeBias(self, threshOut, bitWidth, val):
    return self._lib.XDNNQuantizeBias(threshOut, bitWidth, val)

  def quantizev3Bias(self, threshIn, threshParams, bitWidth, val, doRounding):
    return self._lib.XDNNV3QuantizeBias(threshIn, threshParams, bitWidth, val, doRounding)

  def quantizeTensor(self, threshIn, bitWidth, v):
    return self._lib.XDNNQuantizeTensor(threshIn, bitWidth, v, v.size)
  def quantizeAvgPool(self, sumQuantizedVal, scaleVal, postShiftVal, bitWidth):
    return self._lib.XDNNQuantizeAvgPool(sumQuantizedVal, scaleVal, postShiftVal, bitWidth)

  def unquantizeTensor(self, threshOut, bitWidth, v):
    return self._lib.XDNNUnQuantizeTensor(threshOut, bitWidth, v, v.size)

_xdnnManager = None

def createManager ( libFile ):
  global _xdnnManager
  if not _xdnnManager:
    _xdnnManager = XDNNManager(libFile)
  return True
    
def createHandle(xclbin, kernel="kernelSxdnn_0", libFile=None, numHandles=1):
  """
  Programs a hardware acceleration engine to the FPGA, and initializes communication.
  
  :param xclbin: Path to binary image (a.k.a. xclbin) to be loaded.
  :type xclbin: str.
  :param kernel: Name of kernel in xclbin. Always use "kernelSxdnn_0". To be deprecated.
  :type kernel: str.
  :param libFile: Path to libxfdnn.so shared library. This is the high performance middleware invoked by the Python APIs.
  :type libFile: str.
  :param numHandles: Number of handles to be created. This parameter is reserved for future development.
  :type numHandles: int.
  :returns: int -- Return Code. Expect 0 for success.
  """
  createManager (libFile)
  return _xdnnManager.createHandle(xclbin, kernel, numHandles)

def closeHandle():
  """
  Terminates communication by destroying handle. No return value.
  """
  return _xdnnManager.closeHandle()

def makeFPGAFloatArray(n):
  return _xdnnManager.makeFPGAFloatArray(n)

def v3computeWeightsBiasQuantSize(kWidth, kHeight, outCh, srcFullSectNum, srcReplSectNum, srcReplUnitNum, is8bit):
  return _xdnnManager._lib.XDNNV3ComputeWeightsBiasQuantSize(kWidth, kHeight, outCh, srcFullSectNum, srcReplSectNum, srcReplUnitNum, is8bit)

def computeWeightsBiasQuantSize(kWidth, kHeight, inCh, outCh, doQuant):
  return _xdnnManager._lib.XDNNComputeWeightsBiasQuantSize(\
    kWidth, kHeight, inCh, outCh, doQuant)

def makeWeightsBiasQuantBlob(size):
  return _xdnnManager._lib.XDNNMakeWeightsBiasQuantBlob(size)

def v3fillWeightsBiasQuantBlob(blob, offset, cfgFile, 
  weights, scaleWeight, bias, scaleBias, 
  kw, kh, inch, outch, srcFullSectNum, srcReplSectNum, srcReplUnitNum, srcReplUnitWidth, convHalfRateMode,  layerName = ""):
  return _xdnnManager.v3fillWeightsBiasQuantBlob(\
    blob, offset, cfgFile, 
    weights, scaleWeight, bias, scaleBias, kw, kh, inch, outch, srcFullSectNum, srcReplSectNum, srcReplUnitNum, srcReplUnitWidth, convHalfRateMode, layerName)

def fillWeightsBiasQuantBlob(blob, offset, cfgFile, 
  weights, scaleWeight, bias, scaleBias, 
  kw, kh, inch, outch, layerName = ""):
  return _xdnnManager.fillWeightsBiasQuantBlob(\
    blob, offset, cfgFile, 
    weights, scaleWeight, bias, scaleBias, kw, kh, inch, outch, layerName)

def loadBlobToDdr(blob, size, layer2OffsetMap, PE=-1):
  return _xdnnManager.loadBlobToDdr(blob, size, layer2OffsetMap, PE)

# this is typically the first method that is called.
def prepareInputsForFpga(inputs, cfgFile, scale, PE=-1, layerName = "",streamId=0):
  return _xdnnManager.prepareInputsForFpga( inputs, cfgFile, scale, PE, layerName, streamId)

def passThruInputsForFpga(ctypesArr, batchSize, imgSize, cfgFile, scale, PE=-1, layerName = "", streamId=0):
  return _xdnnManager.passThruInputsForFpga(ctypesArr, batchSize, imgSize, cfgFile, scale, PE, layerName, streamId)

def initScript(netFile, weightsBlob, numBatches, cfgFile, scale, PE, streamId=0):
  """
  Loads the network schedule on the hardware accelerator. This API call is blocking.

  :param netFile: Path to file which contains network specific instructions, this file should be generated by xfdnn compiler.
  :type netFile: str.
  :param weightsBlob: This is an object constructed by the xdnn_io.loadWeights API, which provides the address in memory where weights preside.
  :type weightsBlob: <class 'xdnn.LP_c_short'>.  
  :param numBatches: Number of images to process per execute call.
  :type numBatches: int.
  :param cfgFile: Path to file which contains network specific quantization parameters, this file should be generated by xfdnn quantizer.
  :type cfgFile: str.
  :param scale: Scale used for bias terms in global quantization mode, typically set to 30.
  :type scale: int.
  :param PE: Index used to target specific processing element. Use -1 for autoselect. There can be from 1 to 6 processing elements in a particular xclbin.
  :type PE: int.
  """
  return _xdnnManager.initScript(netFile, weightsBlob, numBatches, cfgFile, scale, PE, streamId)

def execute(netFile, weightsBlob, inputs, output, numBatches, 
  cfgFile, scale, PE=-1, streamId=0):
  """
  Executes inference on the hardware accelerator. This API call is blocking.

  :param netFile: Path to file which contains network specific instructions, this file should be generated by xfdnn compiler.
  :type netFile: str.
  :param weightsBlob: This is an object constructed by the xdnn_io.loadWeights API, which provides the address in memory where weights preside.
  :type weightsBlob: <class 'xdnn.LP_c_short'>.  
  :param inputs: Array holding the input volume for which to run inference. This object is constructed by the xdnn_io.prepareInput API.
  :type inputs: <class 'xdnn.LP_c_short_Array_1'>.
  :param outputs: Array holding the result of the inference ran on the hardware accelerator. Shape will be (fpgaoutsz,) where fpgaoutsz is the total number of elements in the final activation ran in HW.
  :type outputs: numpy.ndarray.
  :param numBatches: Number of images to process per execute call.
  :type numBatches: int.
  :param cfgFile: Path to file which contains network specific quantization parameters, this file should be generated by xfdnn quantizer.
  :type cfgFile: str.
  :param scale: Scale used for bias terms in global quantization mode, typically set to 30.
  :type scale: int.
  :param PE: Index used to target specific processing element. Use -1 for autoselect. There can be from 1 to 6 processing elements in a particular xclbin.
  :type PE: int.
  :param streamId: Argument not required. 
  :type streamId: int.
  :returns: int -- Return Code. Expect 0 for success.
  """
  return _xdnnManager.execute(netFile, weightsBlob, inputs, output, 
    numBatches, cfgFile, scale, PE, streamId, True)

def exec_async (netFile, weightsBlob, inputs, output, numBatches, 
  cfgFile, scale, PE=-1, streamId=0):
  """
  Executes inference on the hardware accelerator. This API call is non-blocking. The result of execution can be fetched using xdnn.get_result.

  :param netFile: Path to file which contains network specific instructions, this file should be generated by xfdnn compiler.
  :type netFile: str.
  :param weightsBlob: This is an object constructed by the xdnn_io.loadWeights API, which provides the address in memory where weights preside.
  :type weightsBlob: <class 'xdnn.LP_c_short'>.  
  :param inputs: Array holding the input volume for which to run inference. This object is constructed by the xdnn_io.prepareInput API.
  :type inputs: <class 'xdnn.LP_c_short_Array_1'>.
  :param outputs: Array holding the result of the inference ran on the hardware accelerator. Shape will be (fpgaoutsz,) where fpgaoutsz is the total number of elements in the final activation ran in HW.
  :type outputs: numpy.ndarray.
  :param numBatches: Number of images to process per execute call.
  :type numBatches: int.
  :param cfgFile: Path to file which contains network specific quantization parameters, this file should be generated by xfdnn quantizer.
  :type cfgFile: str.
  :param scale: Scale used for bias terms in global quantization mode, typically set to 30.
  :type scale: int.
  :param PE: Index used to target specific processing element. Use -1 for autoselect. There can be from 1 to 6 processing elements in a particular xclbin.
  :type PE: int.
  :param streamId: Stream ID used to recover result at a later time. 
  :type streamId: int.
  :returns: int -- Return Code. Expect 0 for success.
  """
  return _xdnnManager.execute (netFile, weightsBlob, inputs, output, 
    numBatches, cfgFile, scale, PE, streamId, False)

def get_result(PE=-1, streamId=0):
  """
  Get result of execution for a given PE, and a given stream. This API is used in conjuntion with xdnn.exec_async.

  :param PE: Index used to target specific processing element. Use -1 for autoselect. There can be from 1 to 6 processing elements in a particular xclbin.
  :type PE: int.
  :param streamId: Stream ID to recover result from. 
  :type streamId: int.
  :returns: int -- Return Code. Expect 0 for success.
  """
  return _xdnnManager.get_result(PE, streamId)

def softmax(x):
  e_x = np.exp(x-np.max(x))
  return (e_x)/(e_x.sum(keepdims=True))

# Old Slow Softmax 
#def softmax(data):
#  import math
#  maxVal = data.max()
#  for x in np.nditer(data, op_flags=['readwrite']):
#    x[...] = math.exp(x - maxVal)
#
#  totalVal = np.sum(data)
#  for x in np.nditer(data, op_flags=['readwrite']):
#    x[...] = x / totalVal
#
#  return data    

def computeSoftmax(data, num):
  """
  Compute the softmax of a given activation or a set of activations.

  :param data: Activation or a set of activations corresponding to multiple images stored as a 1D Array.
  :type data: numpy.ndarray.
  :param num: Number of images processed.
  :type num: int.
  :returns: numpy.ndarray -- Softmax Activation.
  """
  outSize = len(data) / num

  i = 0
  for n in range(num):
    data[i:i+outSize] = softmax(data[i:i+outSize])
    i += outSize

  return data
  
def computeFC(weight, bias, data, M, N, K, useBlas):
  """
  Compute the inner product layer for a given activation or a set of activations. WX+B.
  
  :param weight: Weights corresponding to the inner product layer. These weights are extracted by the xdnn_io.loadWeights API.
  :type weight: numpy.ndarray
  :param bias: Biases corresponding to the inner product layer. These biases are extracted by the xdnn_io.loadWeights API.
  :type bias: numpy.ndarray
  :param data: Activation or a set of activations corresponding to multiple images stored as a 1D Array.
  :type data: numpy.ndarray.
  :param M: Number of inferences being ran in parallel. i.e. # of images. 
  :type M: int.
  :param N: Number of elements in the output volume returned by Inner Product for a single inference. This is specific to the network. 
  :type N: int.
  :param K: Number of elements in the output volume returned by FPGA for a single inference. This is specific to the network. 
  :type K: int.
  :param useBlas: Use CBLAS to accelerate arithmetic in CPU.
  :type useBlas: bool.
  :returns: numpy.ndarray -- Inner Product result.
  """
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
  output += bias

  return output

def quantizeInputs(layerName, cfgFile, scale, inputs):
    quantizedOutput = np.empty( shape = inputs.shape, dtype=np.int16, order='C')
    #np.ascontiguousarray(quantizedOutput, dtype=np.int16)
        
    numBatches = inputs.shape[0]        
    imgSize = np.product ( inputs.shape[1:] )
    
    if inputs.flags['C_CONTIGUOUS'] == False:
        inputs = np.ascontiguousarray(inputs)
   
    numFpgaInputs = _xdnnManager._lib.XDNNPrepareInputFlatArray(\
      layerName, inputs, quantizedOutput,
      numBatches, imgSize, cfgFile, scale)

    return quantizedOutput[:numFpgaInputs]
  
def readWeightsFile(fname):
  return _xdnnManager.readWeightsFile(fname)

#from multiprocessing import sharedctypes, Manager
#g_sharedMemManager = Manager()
#g_sharedMemDict = g_sharedMemManager.dict()
#
#def getOrCreateNumpyCtypesArr(key, n, dtype):
#  arr = None 
#
#  if key in g_sharedMemDict:
#    arr = g_sharedMemDict[key]
#  else:
#    ct = None
#    if dtype == np.int16:
#      ct = c_short
#
#    arr = sharedctypes.RawArray(ct, n)
#    g_sharedMemDict[key] = arr
#
#  nparr = np.frombuffer(arr, dtype)
#
#  return (nparr, arr)
