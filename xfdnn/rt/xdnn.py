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

class XDNNFPGAOp:
  def __init__ (self, handles, args):
    libFile = os.environ["LIBXDNN_PATH"]
    if not libFile or not os.path.isfile(libFile):
      raise AssertionError("XDNN library .so file %s not found" % libFile)

    self._libFile = os.path.abspath(libFile)
    self._lib = cdll.LoadLibrary(self._libFile)
    self._handles = handles

    funcMap = {} # "external name -> lib name"
    funcMap["v3computeWeightsBiasQuantSize"] = "XDNNV3ComputeWeightsBiasQuantSize"
    funcMap["computeWeightsBiasQuantSize"] = "XDNNComputeWeightsBiasQuantSize"
    funcMap["makeWeightsBiasQuantBlob"] = "XDNNMakeWeightsBiasQuantBlob"

    for k in funcMap:
      v = funcMap[k]
      setattr(self, k, getattr(self._lib, v))

    self._lib.xHostMalloc.argtypes = [c_size_t]
    self._lib.xHostMalloc.restype = c_void_p
    self._lib.xMalloc.argtypes \
      = [c_void_p , c_size_t, c_bool]
    self._lib.xMalloc.restype = c_void_p
    self._lib.xMemcpyHost2Device.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t]
    self._lib.xFree.argtypes = [c_void_p, c_void_p, c_bool]
    self._lib.xblasLoadA.argtypes = [c_void_p, c_int, c_void_p, c_char_p, c_void_p, c_int]
    self._lib.XDNNMakeWeightsBiasQuantBlob.argtypes = [c_int]
    self._lib.XDNNWaitForResults.argtypes = [c_void_p, c_int]
    self._lib.XDNNMakeWeightsBiasQuantBlob.restype = POINTER(c_short)
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

    self._lib.XDNNReadWeightsFile.argtypes \
      = [c_char_p, POINTER(POINTER(c_char)),
         POINTER(POINTER(c_int)), POINTER(POINTER(c_int)),
         POINTER(POINTER(c_int)), POINTER(POINTER(c_int)),
         POINTER(POINTER(c_int)), POINTER(POINTER(c_float))]

    self._lib.XDNNMakeScriptExecutor.argtypes \
      = [POINTER(c_void_p), c_int,
         POINTER(c_short), c_char_p, c_char_p, c_float,
         c_int]
    self._lib.XDNNMakeScriptExecutor.restype = c_void_p
    self._lib.XDNNExecute_1D_float.argtypes = [c_void_p,
                                      np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"), np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                      c_uint, c_uint, c_int, c_bool]

    self._lib.XDNNExecute_2D_float.argtypes = [c_void_p,
                                      POINTER(POINTER(c_float)), np.ctypeslib.ndpointer(c_float, flags="C_CONTIGUOUS"),
                                      c_int, c_int, c_bool]
    self._qInput = np.empty(((args['batch_sz'],) + args['in_shape']), dtype=np.int16, order='C')

    self._weights = self.loadWeights(args)
    self._args = args

    numHandles = len(handles)
    handlePtrs = (c_void_p*numHandles)()
    for i,h in enumerate(self._handles):
      handlePtrs[i] = h

    self._executor = self._lib.XDNNMakeScriptExecutor(\
      handlePtrs, numHandles, self._weights, c_char_p(args['netcfg'].encode('utf-8')), c_char_p(args['quantizecfg'].encode('utf-8')),
      args['scaleB'], self.getMask(args['PE']))

  def loadBlobToDdr(self, blob, size, layer2OffsetMap, PE=-1):
    if not isinstance (PE, list): PE = [PE]

    numBytes = size * 2
    fps = []
    for h in self._handles:
      fp = self._lib.xMalloc(h, numBytes, True)
      self._lib.xMemcpyHost2Device(h, blob, fp, numBytes)
      fps.append((h, fp))

    for peIdx in PE:
      for h in self._handles:
        self._lib.xblasLoadA(h, size, blob, c_char_p(layer2OffsetMap.encode('utf-8')), None, peIdx)

    return fps

  def parseCompilerFileJson(self, compilerFileName):
    with open(compilerFileName) as json_data:
      compilerContent = json.load(json_data)
    allLayersParams=[]
    layerParams={}
    for i in range(len(compilerContent["network"])):
      layerObj = compilerContent['network'][i]['xdnn_kv']
      if bool(layerObj):
        if layerObj["XNOp"] == "XNConv" \
          or layerObj["XNOp"] == "XNConvDepth": 
          layerParams={}
          layerParams['name']=layerObj["name"]
          xdnnFields = {'kernW': 'kernel_w', 
                        'kernH': 'kernel_h', 
                        'inChans': 'inchan', 
                        'outChans': 'outchan',
                        'srcFullSectNum': 'src_full_sect_num', 
                        'srcReplSectNum': 'src_repl_sect_num', 
                        'srcReplUnitNum': 'src_repl_unit_num', 
                        'srcReplUnitWidth': 'src_repl_unit_width', 
                        'convHalfRateMode': 'en_halfrate_mode'}
          for (k, v) in xdnnFields.iteritems():
            if v in layerObj:
              layerParams[k] = int(layerObj[v])

          allLayersParams.append(layerParams)
        elif layerObj["XNOp"] == "XNMaxPoolPipelined":
          layerParams={}
          layerParams['name']=layerObj["conv_name"]
          xdnnFields = {'kernW': 'conv_kernel_w', 
                        'kernH': 'conv_kernel_h', 
                        'inChans': 'inchan', 
                        'outChans': 'pool_inchan',
                        'srcFullSectNum': 'src_full_sect_num', 
                        'srcReplSectNum': 'src_repl_sect_num', 
                        'srcReplUnitNum': 'src_repl_unit_num', 
                        'srcReplUnitWidth': 'src_repl_unit_width', 
                        'convHalfRateMode': 'en_halfrate_mode'}
          for (k, v) in xdnnFields.iteritems():
            if v in layerObj:
              layerParams[k] = int(layerObj[v])
    
          allLayersParams.append(layerParams)

    return allLayersParams

  def parseCompilerFile(self,compilerFileName):
    if 'json' in compilerFileName:
      return self.parseCompilerFileJson(compilerFileName)

    with open(compilerFileName) as compilerReadStream:
      compilerContent = compilerReadStream.readlines()
    compilerContent = [x.strip().split(" ") for x in compilerContent]
    allLayersParams=[]
    layerParams={}
    for i in range(len(compilerContent)):
      if compilerContent[i][1] == "XNConv":
        layerParams={}
        layerParams['name']=compilerContent[i][2]
        layerParams['kernW']=int(compilerContent[i][3])
        layerParams['kernH']=int(compilerContent[i][4])
        layerParams['inChans']=int(compilerContent[i][19])
        layerParams['outChans']=int(compilerContent[i][23])
        if len(compilerContent[i]) >= 47:
          layerParams['srcFullSectNum']=int(compilerContent[i][25])
          layerParams['srcReplSectNum']=int(compilerContent[i][26])
          layerParams['srcReplUnitNum']=int(compilerContent[i][27])
          layerParams['srcReplUnitWidth']=int(compilerContent[i][28])
          layerParams['convHalfRateMode']=int(compilerContent[i][47])

        allLayersParams.append(layerParams)
      elif compilerContent[i][1] == "XNMaxPoolPipelined":
        layerParams={}
        layerParams['name']=compilerContent[i][47]
        layerParams['kernW']=int(compilerContent[i][48])
        layerParams['kernH']=int(compilerContent[i][49])
        layerParams['inChans']=int(compilerContent[i][12])
        layerParams['outChans']=int(compilerContent[i][60])
        layerParams['srcFullSectNum']=int(compilerContent[i][17])
        layerParams['srcReplSectNum']=int(compilerContent[i][18])
        layerParams['srcReplUnitNum']=int(compilerContent[i][19])
        layerParams['srcReplUnitWidth']=int(compilerContent[i][20])
        layerParams['convHalfRateMode']=int(compilerContent[i][39])

        allLayersParams.append(layerParams)

    return allLayersParams

  def loadWeights(self, args) :
    """
    Load weights to off chip device memory. The weights are first quantized.

    :param args: Collection of arguments. Most importanly args["datadir"] which is the path to a folder containing weights & biases.
    :type args: dict.
    :returns: tuple -- (weightsBlob, fcWeight, fcBias) -- <class 'xdnn.LP_c_short'>, numpy.ndarray, numpy.ndarray
    """
    return self.loadWeightsBiasQuant(args)

  def _loadLayerParamsFromFiles(self, args):
    xdnnv3 = 'xdnnv3' in args and args['xdnnv3']
    compilerParamsList = self.parseCompilerFile(args['netcfg'])
    compilerParams = { lp['name']: lp for lp in compilerParamsList }

    paramsFromDataDir = {}
    if args['datadir']:
      # collect params from files
      fi = 0
      while True:
        fname = "%s/fwbqb_%d" % (args['datadir'], fi)
        if not os.path.isfile(fname):
          break

        with open(fname, 'r') as f:
          data = f.read()
          vals = data.strip().split(' ')
          layerName = vals[0]
          if 'v2WeightsFormat' in args and args['v2WeightsFormat'] == 1:
            kernWidth  = int(vals[1])
            kernHeight = int(vals[2])
            inChans    = int(vals[3])
            outChans   = int(vals[4])
            weights = [float(v) for v in vals[5:]]
          else:
            kernWidth = kernHeight = int(vals[1])
            inChans   = int(vals[2])
            outChans  = int(vals[3])
            weights = [float(v) for v in vals[4:]]

        fname = "%s/fwbqb_bias_%d" % (args['datadir'], fi)
        with open(fname, 'r') as f:
            data = f.read()
            vals = data.strip().split(' ')
            if 'v2WeightsFormat' in args and args['v2WeightsFormat'] == 1:
              vals = vals[5:]
            else:
              vals = vals[4:]
            bias = [float(v) for v in vals]

        paramsFromDataDir[layerName] = { \
          'layerName': layerName,
          'kernWidth': kernWidth,
          'kernHeight': kernHeight,
          'inChans': inChans,
          'outChans': outChans,
          'weights': weights,
          'bias': bias }
        fi += 1
    visitedLayers = set()
    layerParams = []
    for l in compilerParamsList:
      layerName  = l['name']
      layerName  = layerName.split("#", 1)[0] # to handle partial layers
      if layerName in visitedLayers:
        continue
      visitedLayers.add(layerName)
      kernWidth  = l['kernW']
      kernHeight = l['kernH']
      inChans    = l['inChans']
      outChans   = l['outChans']
      weights    = None
      bias       = None

      if layerName in paramsFromDataDir:
        dParams = paramsFromDataDir[layerName]
        assert kernWidth == dParams['kernWidth']
        assert kernHeight == dParams['kernHeight']
        assert inChans == dParams['inChans']
        assert outChans == dParams['outChans']
        weights = dParams['weights']
        bias = dParams['bias']
      else:
        # no datadir provided; load dummy weights
        print("WARNING: loading dummy weights for %s" % layerName)
        weightsSize = kernWidth * kernHeight * inChans * outChans
        weights = [0] * weightsSize
        bias = [0] * outChans

      layerParam = {
        "name": layerName,
        "kern_w": kernWidth,
        "kern_h": kernHeight,
        "in_ch": inChans,
        "out_ch": outChans,
        "weights": weights,
        "bias": bias,
        "quantize": True if args['quantizecfg'] else False
      }

      if xdnnv3:
        # append compiler params
        compilerParam = compilerParams[layerName]
        cparams = ['srcFullSectNum', 'srcReplSectNum', 'srcReplUnitNum',
          'srcReplUnitWidth', 'convHalfRateMode']
        for cp in cparams:
          layerParam[cp] = compilerParam[cp]

        # sanity checks
        assert kernWidth == compilerParam['kernW']
        assert kernHeight == compilerParam['kernH']
        assert inChans == compilerParam['inChans']

      layerParams.append(layerParam)

    return layerParams

  def is8BitMode(self, args):
    with open(args['xclbin']+'.json') as f:
      obj = json.load(f)
      if 'XDNN_BITWIDTH' not in obj:
        return False
      bitwidth = int(obj['XDNN_BITWIDTH'])
      if bitwidth == 8:
        return True

    return False

  def loadWeightsBiasQuant(self, args):
      print("Loading weights/bias/quant_params to FPGA...")

      if '_layerParams' in args:
        layerParams = args['_layerParams']
      else:
        layerParams = self._loadLayerParamsFromFiles(args)
      xdnnv3 = 'xdnnv3' in args and args['xdnnv3']
      is8bit = self.is8BitMode(args)

      weightLayerParams = []
      for lp in layerParams:
        if lp["weights"]:
          weightLayerParams.append(lp)

      size = 0
      for lp in weightLayerParams:
        if xdnnv3:
          size += self.v3computeWeightsBiasQuantSize(\
            lp['kern_w'], lp['kern_h'], lp['out_ch'],
            lp['srcFullSectNum'], lp['srcReplSectNum'],
            lp['srcReplUnitNum'], is8bit) * 2
        else:
          size += self.computeWeightsBiasQuantSize(\
            lp['kern_w'], lp['kern_h'],
            lp['in_ch'], lp['out_ch'], lp['quantize'])

      blob = self.makeWeightsBiasQuantBlob(size)

      layer2OffsetMapStr = ""
      offset = 0
      for i, lp in enumerate(weightLayerParams):
        if layer2OffsetMapStr != "":
          layer2OffsetMapStr += ","
        layer2OffsetMapStr += "%s:%d" % (lp['name'], offset)

        if xdnnv3:
          offset += self.v3fillWeightsBiasQuantBlob(blob, offset,
            args['quantizecfg'], lp['weights'], args['scaleA'],
            lp['bias'], args['scaleB'], lp['kern_w'], lp['kern_h'],
            lp['in_ch'], lp['out_ch'],
            lp['srcFullSectNum'], lp['srcReplSectNum'],
            lp['srcReplUnitNum'], lp['srcReplUnitWidth'],
            lp['convHalfRateMode'], lp['name'])
        else:
          offset += self.fillWeightsBiasQuantBlob(blob, offset,
            args['quantizecfg'], lp['weights'], args['scaleA'],
            lp['bias'], args['scaleB'], lp['kern_w'], lp['kern_h'],
            lp['in_ch'], lp['out_ch'], lp['name'])

      self.loadBlobToDdr(blob, size, layer2OffsetMapStr, int(args['PE']))

      return blob

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
      blob, offset, c_char_p(layerName.encode('utf-8')),
      c_char_p(cfgFile.encode('utf-8')),
      cWeights, len(cWeights), scaleWeight,
      cBias, len(cBias), scaleBias,
      c_ushort(kw), c_ushort(kh), inch, outch)

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

  def getMask(self, peList):
    if not isinstance(peList, list): peList = [peList]

    peMask = 0
    for peId in peList:
      if peId == -1: return 0
      peMask = peMask | (1 << peId)
    return peMask

  def execute(self, inputs, output, streamId=0, blocking=True ):
    """
    Executes inference on the hardware accelerator. This API call is blocking.

    :param inputs: Array holding the input volume for which to run inference.
    :type inputs: numpy array or array of raw c_short pointers.
    :param outputs: Array holding the result of the inference ran on the hardware accelerator. Shape will be (fpgaoutsz,) where fpgaoutsz is the total number of elements in the final activation ran in HW.
    :type outputs: numpy.ndarray.
    :param streamId: Argument not required.
    :type streamId: int.
    """
    if isinstance(inputs,np.ndarray):
      if inputs.dtype == np.float32:
        self._lib.XDNNExecute_1D_float(self._executor,
                                    inputs, output, inputs.shape[0], np.product(inputs.shape[1:]), streamId, blocking)        
      else:
        raise ValueError( "Unsupported input datatype", inputs.dtype)

    else:
      pointer_ar = (POINTER(c_float) * len(inputs) )(*inputs)
      self._lib.XDNNExecute_2D_float ( self._executor, pointer_ar, output, len(inputs), streamId, blocking )

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

    self._lib.XDNNFetchBatchBlob.argtypes \
      = [POINTER(c_short), c_int, c_char_p]
    self._lib.XDNNStd2XdnnV3.argtypes = [np.ctypeslib.ndpointer(c_short, flags="C_CONTIGUOUS"), POINTER(c_short), c_int, c_int, c_int, c_bool, c_int]

    self._lib.XDNNGetV3InputFormatSize.argtypes = [c_int, c_int, c_int]
    self._lib.XDNNGetV3InputFormatSize.restype = c_int
    self._lib.XDNNGetHostDeviceName.argtypes = [c_char_p]
    self._lib.XDNNGetHostDeviceName.restype = c_char_p

    self._exposeLibFunctions()

  def _exposeLibFunctions(self):
    funcMap = {} # "external name -> lib name"
    funcMap["quantizeBias"] = "XDNNQuantizeBias"
    funcMap["quantizev3Bias"] = "XDNNV3QuantizeBias"
    funcMap["quantizeAvgPool"] = "XDNNQuantizeAvgPool"
    funcMap["fetchbatchblob"] = "XDNNFetchBatchBlob"
    funcMap["std2xdnnv3"] = "XDNNStd2XdnnV3"
    funcMap["getV3InputFormatSize"] = "XDNNGetV3InputFormatSize"
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

  def computeFC(self, weight, bias, data, M, N, K, out):
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
    :param out: Inner Product result (output volume)
    :type out: numpy.ndarray.
    """
    M = int(M)
    N = int(N)
    K = int(K)
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
