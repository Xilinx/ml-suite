#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from __future__ import print_function

from ctypes import *
import json
import os, sys
import timeit
import numpy as np
from multiprocessing.managers import BaseManager
#Parsing JSON directly is easier than passing all the necessary params from C++ to python
#Pybind11 will make passing data between C++/python easier and will remove the need for this class
class CompilerJsonParser:
  def __init__(self, compilerJSONFile):
    self._jsonObj = None
    inName = {}
    outName = {}

    self._inputs = {}
    self._outputs = {}
    with open(compilerJSONFile) as f:
      self._jsonObj = json.load(f)

      for i in self._jsonObj["inputs"]:
        inName[i["input_name"]] = i["input_name"]

      for i in self._jsonObj["outputs"]:
        outName[ i["output_name"] ] = i["previous_tensors"][0]
        #print ( i["previous_tensors"][0] , " -> ", i["output_name"] )

      for i in self._jsonObj["network"]:
        if i["name"] in inName.keys():
          self._inputs[ inName[ i["name"] ] ] = i["outputshapes"]

        elif i["name"] in outName.keys():
          self._outputs[ outName[ i["name"] ] ] = i["outputshapes"]
          #self._outputs[ i["name"] ] = i["outputshapes"]
        # mergedItem
     #   if "merged" in i:
     #     for mergedItem in i["merged"]:
            #print ( "mergeditem ", mergedItem , " -> ", i["name"] )
            #self._hwoutname[mergedItem] = i["name"]


  def getInputs(self):
    return self._inputs

  def getOutputs(self):
    return self._outputs

class XDNNFPGAOp:
  def __init__ (self, handles, args):
    libFile = os.environ["LIBXDNN_PATH"]
    if not libFile or not os.path.isfile(libFile):
      raise AssertionError("XDNN library .so file %s not found" % libFile)

    self._libFile = os.path.abspath(libFile)
    self._lib = cdll.LoadLibrary(self._libFile)
    self._handles = handles

    self._prev_time = 0.0

    funcMap = {} # "external name -> lib name"
    funcMap["v3computeWeightsBiasQuantSize"]  = "XDNNV3ComputeWeightsBiasQuantSize"
    funcMap["computeWeightsBiasQuantSize"]    = "XDNNComputeWeightsBiasQuantSize"
    funcMap["makeWeightsBiasQuantBlob"]       = "XDNNMakeWeightsBiasQuantBlob"


    for k, func in funcMap.items():
      setattr(self, k, getattr(self._lib, func))

    self._lib.XDNNMakeScriptExecutor.argtypes \
      = [POINTER(c_void_p), c_int,
         POINTER(c_short), c_char_p, c_char_p, c_float,
         c_int]
    self._lib.XDNNMakeScriptExecutorAndLoadWeights.argtypes \
      = [POINTER(c_void_p), c_int,
         c_char_p, c_char_p, c_char_p, c_float,
         c_int]
    self._lib.XDNNMakeScriptExecutorAndLoadWeightsFromMem.argtypes \
      = [POINTER(c_void_p), c_int, c_int, POINTER(c_char_p),
         POINTER(POINTER(c_float)), POINTER(c_int),
         POINTER(POINTER(c_float)), POINTER(c_int),
         c_char_p, c_char_p, c_float, c_int]
    self._lib.XDNNMakeScriptExecutor.restype = c_void_p
    self._lib.XDNNMakeScriptExecutorAndLoadWeights.restype = c_void_p
    self._lib.XDNNMakeScriptExecutorAndLoadWeightsFromMem.restype = c_void_p

    self._lib.XDNNExecute_2D_float.argtypes = [c_void_p,
                                      POINTER(POINTER( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS") )),
                                      POINTER(c_char_p),
                                      c_uint,
                                      POINTER( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")),
                                      POINTER(c_uint),
                                      POINTER(c_char_p),
                                      c_uint,
                                      c_uint,
                                      c_int,
                                      c_bool]

    self._lib.XDNNSetCustomStartIdx.argtypes = [c_void_p, c_int, c_int]
    self._lib.XDNNSetCustomStopIdx.argtypes = [c_void_p, c_int, c_int]
    self._lib.XDNNReadHardwareCounter.argtypes = [c_void_p, c_int, c_int]
    self._lib.XDNNReadHardwareCounter.restype = c_float
    self._args = args

    numHandles = len(handles)
    handlePtrs = (c_void_p*numHandles)()
    for i,h in enumerate(self._handles):
      handlePtrs[i] = h

    if '_layerParams' not in args:
      # load from disk
      self._executor = self._lib.XDNNMakeScriptExecutorAndLoadWeights(\
        handlePtrs, numHandles,
        c_char_p(args['weights']),
        c_char_p(args['netcfg'].encode('utf-8')),
        c_char_p(args['quantizecfg'].encode('utf-8')),
        args['scaleB'], self.getMask(args['PE']))
    else:
      # load directly from mem
      layerParams = self._args['_layerParams']

      weightLayerIndices = []
      for i, lp in enumerate(layerParams):
        if lp["weights"]:
          lp["weights_sz"] = len(lp["weights"])
          lp["weights"] = np.ascontiguousarray(lp["weights"], dtype=np.float32).flatten()
          lp["bias_sz"] = len(lp["bias"])
          lp["bias"] = np.ascontiguousarray(lp["bias"], dtype=np.float32).flatten()
          weightLayerIndices.append(i)

      numWeightLayers = len(weightLayerIndices)
      weightLayerNames = (c_char_p * numWeightLayers)()
      weights = (POINTER(c_float) * numWeightLayers)()
      bias = (POINTER(c_float) * numWeightLayers)()
      weightsSz = (c_int * numWeightLayers)()
      biasSz = (c_int * numWeightLayers)()

      for i, idx in enumerate(weightLayerIndices):
        lp = layerParams[idx]
        weightLayerNames[i] = lp['name']
        weights[i] = lp['weights'].ctypes.data_as(POINTER(c_float))
        weightsSz[i] = lp['weights_sz']
        bias[i] = lp['bias'].ctypes.data_as(POINTER(c_float))
        biasSz[i] = lp['bias_sz']

      self._executor \
        = self._lib.XDNNMakeScriptExecutorAndLoadWeightsFromMem(\
        handlePtrs, numHandles, numWeightLayers,
        weightLayerNames, weights, weightsSz, bias, biasSz,
        c_char_p(args['netcfg'].encode('utf-8')),
        c_char_p(args['quantizecfg'].encode('utf-8')),
        args['scaleB'], self.getMask(args['PE']))

    self._compilerJSONObj = CompilerJsonParser( args['netcfg'] )
    self._npInputs = {}
    self._npOutputs = {}

    for k, v in self._compilerJSONObj.getInputs().items():
      if isinstance(args['batch_sz'], dict):
        batch_size = args['batch_sz'][k]
      else:
        batch_size = args['batch_sz']
      self._npInputs[k] = np.empty(((batch_size,) + tuple(v[1:])), dtype=np.float32, order='C')

    for k, v in self._compilerJSONObj.getOutputs().items():
      self._npOutputs[k] = np.empty(((batch_size,) + tuple(v[1:])), dtype=np.float32, order='C')

  def getInputDescriptors(self):
    return self._compilerJSONObj.getInputs()

  def getOutputDescriptors(self):
    return self._compilerJSONObj.getOutputs()

  def getOutputs(self):
    return self._npOutputs

  def getInputs(self):
    return self._npInputs

  def is8BitMode(self, args):
    with open(args['xclbin']+'.json') as f:
      obj = json.load(f)
      if 'XDNN_BITWIDTH' not in obj:
        return False
      bitwidth = int(obj['XDNN_BITWIDTH'])
      if bitwidth == 8:
        return True

    return False

  def getMask(self, peList):
    if not isinstance(peList, list): peList = [peList]

    peMask = 0
    for peId in peList:
      if peId == -1: return 0
      peMask = peMask | (1 << peId)
    return peMask

  def execute(self, inputs, outputs, streamId=0, blocking=True ):
    """
    Executes inference on the hardware accelerator. This API call is blocking.

    :param inputs: Array holding the input volume for which to run inference.
    :type inputs: numpy array or array of raw c_short pointers.
    :param outputs: Array holding the result of the inference ran on the hardware accelerator. Shape will be (fpgaoutsz,) where fpgaoutsz is the total number of elements in the final activation ran in HW.
    :type outputs: numpy.ndarray.
    :param streamId: Argument not required.
    :type streamId: int.
    """
    inKeys = inputs.keys()
    outKeys = outputs.keys()

#      for key,val in inputs.iteritems():
#        inKeys.append(key)
#        #if val.flags['C_CONTIGUOUS'] == False:
#        #  raise ValueError( "Input for ", key, " must be C Contiguous" )
#        if isinstance(val,np.ndarray):
#          inSz.append( np.product(val.shape) )
#        elif isinstance( val, list ):
#          inSz.append(np.product(val[0].shape))
#        else:
#          raise ValueError( "Unsupported input format", type(val))
#
#      for key,val in output.iteritems():
#        #if val.flags['C_CONTIGUOUS'] == False:
#        #  raise ValueError( "Output for ", key, " must be C Contiguous" )
#        outKeys.append(key)
#
#        if isinstance(val,np.ndarray):
#          outSz.append( np.product(val.shape) )
#        elif isinstance( val, list ):
#          outSz.append(np.product(val[0].shape))
#        else:
#          raise ValueError( "Unsupported input format", type(val))

    in_name_arr = (c_char_p * len(inKeys) )(*inKeys)
    #in_bufsz_arr = ( c_uint * len(inSz))(*inSz)
    out_name_arr = (c_char_p * len(outKeys) )()
    #out_bufsz_arr = ( c_uint * len(outSz))(*outSz)
    in_ptr = {}
    firstInput = next(iter(inputs.itervalues()))
    if isinstance(firstInput,np.ndarray):
      bsz = firstInput.shape[0]
      for key, array in inputs.iteritems():
        in_ptr[key] = []
        for b in range(bsz):
          in_ptr[key].append ( array[b,...] )
    else:
      in_ptr = inputs

    in_batch = next(iter(in_ptr.itervalues()))

    bsz = len(in_batch)
    ptr_inarr_2d = (POINTER( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS") ) * len(inputs) )()
    i = 0
    for v in in_ptr.itervalues():
      ptr_inarr_2d[i] = ( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS")  * len(v) )()
      for p, p_val in enumerate(v):
        ptr_inarr_2d[i][p] = p_val.ctypes.data_as( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS") )
      i += 1

    ptr_outarr_2d = ( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS") * len(outputs) )()
    out_bufsz_arr = ( c_uint * len(outputs))()

    i = 0
    for k,v in outputs.iteritems():
      ptr_outarr_2d[i] = v.ctypes.data_as( np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS") )
      out_bufsz_arr[i] = np.prod(v.shape[1:])
      out_name_arr[i] = k
      i += 1

    self._lib.XDNNExecute_2D_float ( self._executor, ptr_inarr_2d, in_name_arr,
                                     len(inputs), ptr_outarr_2d, out_bufsz_arr, out_name_arr, len(outputs), bsz, streamId, blocking)

  def exec_async(self, inputs, output, streamId=0):
    """
    Executes inference on the hardware accelerator. This API call is non-blocking. The result of execution can be fetched using xdnn.get_result.

    :param inputs: Array holding the input volume for which to run inference.
    :type inputs: <class 'xdnn.LP_c_short_Array_1'>.
    :param outputs: Array holding the result of the inference ran on the hardware accelerator. Shape will be (fpgaoutsz,) where fpgaoutsz is the total number of elements in the final activation ran in HW.
    :type outputs: numpy.ndarray.
    :param streamId: Stream ID used to recover result at a later time.
    :type streamId: int.
    :returns: int -- Return Code. Expect 0 for success.
    """
    return self.execute(inputs, output, streamId, False)

  def get_result(self, streamId=0):
    """
    Get result of execution for a given PE, and a given stream. This API is used in conjuntion with xdnn.exec_async.
    :param streamId: Stream ID to recover result from.
    :type streamId: int.
    :returns: int -- Return Code. Expect 0 for success.
    """
    return self._lib.XDNNWaitForResults( self._executor, streamId )

  def set_start_idx(self, mbIdx, dflIdx=-1):
    return self._lib.XDNNSetCustomStartIdx(self._executor, dflIdx, mbIdx)

  def set_stop_idx(self, mbIdx, dflIdx=-1):
    return self._lib.XDNNSetCustomStopIdx(self._executor, dflIdx, mbIdx)

  def get_exec_time(self, devIdx=0,cuIdx=0):
    curr_time = self._lib.XDNNReadHardwareCounter(self._executor, devIdx, cuIdx)
    elapsed = curr_time - self._prev_time
    self._prev_time = curr_time
    return elapsed

class XDNNManager:
  def __init__(self, libFile=None):
    if not libFile and "LIBXDNN_PATH" in os.environ:
      libFile = os.environ["LIBXDNN_PATH"]
    if not libFile or not os.path.isfile(libFile):
      raise AssertionError("XDNN library .so file %s not found" % libFile)

    self._libFile = os.path.abspath(libFile)
    self._handles = None
    self._execData = {}

    self._lib = cdll.LoadLibrary(self._libFile)

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
    self._lib.computeSoftmax.argtypes = [np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), c_uint, c_uint ]

    self._lib.XDNNGetHostDeviceName.argtypes = [c_char_p]
    self._lib.XDNNGetHostDeviceName.restype = c_char_p

    self._exposeLibFunctions()

  def _exposeLibFunctions(self):
    funcMap = {} # "external name -> lib name"
    funcMap["quantizeBias"] = "XDNNQuantizeBias"
    funcMap["quantizev3Bias"] = "XDNNV3QuantizeBias"
    funcMap["quantizeAvgPool"] = "XDNNQuantizeAvgPool"
    funcMap["getHostDeviceName"] = "XDNNGetHostDeviceName"

    for k in funcMap:
      v = funcMap[k]
      setattr(self, k, getattr(self._lib, v))

  def createHandle(self, xclbin, kernel="kernelSxdnn_0", handleList = [0]):
    """
    Programs a hardware acceleration engine to the FPGA, and initializes communication.

    :param xclbin: Path to binary image (a.k.a. xclbin) to be loaded.
    :type xclbin: str.
    :param kernel: Name of kernel in xclbin. Always use "kernelSxdnn_0". To be deprecated.
    :type kernel: str.
    :param handleList: List of device ids to create handles for
    :type handleList: list.
    :returns: int -- Return Code. Expect 0 for success.
    """
    ret = 0
    self._handles = []
    for i in range(len(handleList)):
      self._handles.append(c_void_p())
      ret |= self._lib.xblasCreate(
        pointer(self._handles[i]),
        c_char_p(xclbin.encode('utf-8')),
        c_char_p(kernel.encode('utf-8')),
        handleList[i])

    return ret, self._handles

  def closeHandle(self):
    """
    Terminates communication by destroying handle. No return value.
    """
    if not self._handles:
      return

    for h in self._handles:
      self._lib.xblasDestroy(h)

  def quantizev3InterLayer(self, preShift, scale, postShift, bitWidth, v, bias):
    return self._lib.XDNNV3QuantizeInterLayer(\
      preShift, scale, postShift, bitWidth, v, v.size, bias)

  def quantizeInterLayer(self, preShift, scale, postShift, bitWidth, v):
    return self._lib.XDNNQuantizeInterLayer(\
      preShift, scale, postShift, bitWidth, v, v.size)

  def quantizeWeights(self, thresh, bitWidth, v):
    return self._lib.XDNNQuantizeWeights(thresh, bitWidth, v, v.size)

  def quantizeTensor(self, threshIn, bitWidth, v):
    return self._lib.XDNNQuantizeTensor(threshIn, bitWidth, v, v.size)

  def unquantizeTensor(self, threshOut, bitWidth, v):
    return self._lib.XDNNUnQuantizeTensor(threshOut, bitWidth, v, v.size)

  def computeSoftmax(self, data):
    """
    Compute the softmax of a given activation or a set of activations.

    :param data: Activation or a set of activations corresponding to multiple images stored as a 1D Array.
    :type data: numpy.ndarray.
    :param num: Number of images processed.
    :returns: numpy.ndarray -- Softmax Activation.
    """
    for i in range(data.shape[0]):
      self._lib.computeSoftmax(data[i,:], 1, np.prod(data.shape[1:]))
    return data

  def computeFC(self, weight, bias, data,out):
    """
    Compute the inner product layer for a given activation or a set of activations. WX+B.

    :param weight: Weights corresponding to the inner product layer. These weights are extracted by the xdnn_io.loadWeights API.
    :type weight: numpy.ndarray
    :param bias: Biases corresponding to the inner product layer. These biases are extracted by the xdnn_io.loadWeights API.
    :type bias: numpy.ndarray
    :param data: Activation or a set of activations corresponding to multiple images stored as a 1D Array.
    :type data: numpy.ndarray.
    :param out: Inner Product result (output volume)
    :type out: numpy.ndarray.
    """
    M = int(data.shape[0])
    N = int(out.shape[1])
    K = np.product ( data.shape[1:])
    if len(weight) != K*N:
      raise Exception('FC weight dim mismatch')
    if np.size(data) != M*K:
      raise Exception('FC input dim mismatch')
    if len(bias) != N:
      raise Exception('FC bias dim mismatch')

    self._lib.computeFC(weight, bias, data, M, N, K, out)

class XDNNMPManager(BaseManager):
  pass

XDNNMPManager.register('XDNNManager', XDNNManager)

_xdnnManager = None

def createManager ( libFile=None ):
  global _xdnnManager
  if not _xdnnManager \
    or (libFile != None and os.path.abspath(libFile) != _xdnnManager._libFile):
    _xdnnManager = XDNNManager(libFile)
    _exposeXdnnFunctions(_xdnnManager)
  return True

def _exposeXdnnFunctions(xdnnObj):
  #existingNames = dir(sys.modules[__name__])
  xdnnNames = dir(xdnnObj)

  for n in xdnnNames:
    if not n.startswith("_"):
      globals()[n] = getattr(xdnnObj, n)

try:
  # try to load XDNNManager with defaults
  # (picks up libxfdnn.so path from ENV)
  createManager()
except Exception as e:
  print(e)
