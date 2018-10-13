#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from __future__ import print_function 

import argparse
import collections
import ctypes
import json
import os.path
import math, sys
import time, timeit
import xdnn, xdnn_io
import numpy as np
from multiprocessing import Process, Queue, Manager, sharedctypes
import cv2
import PyTurboJPEG

# tqdm used to generate graphical view of latency
# however the current implementation slows down performance
# So do not use for now
#from tqdm import tqdm

# Doesn't work
#import gnuplotlib as gp

# import matplotlib.pyplot as plt

g_doQuant = False
g_fpgaCfgFile = ""
g_scaleA = 10000
g_scaleB = 30
img_shape = (224,224,3)
g_img_shape = [3, 224, 224]
g_raw_scale = 255.0
mean = [104.007, 116.669, 122.679] # BGR for Caffe
g_mean = np.zeros(img_shape,dtype=np.float32)
g_input_scale = 1.0 # 1.0 for GoogLeNet and ResNet, 0.017 for MobileNet

# Broadcast mean across channels
g_mean[...,0] = mean[0]
g_mean[...,1] = mean[1]
g_mean[...,2] = mean[2]

g_xdnnTestDataDir = "data/googlenet_v1"
g_fpgaOutputSize = 1024
g_outputSize = 1000
g_firstFpgaLayerName = "conv1/7x7_s2"
g_labels = None
g_goldenFile = None
g_xclbin = "kernel.xclbin"
g_netFile = "googlenet.fpgaaddr.64.txt"
g_xdnnLib = "libxblas.so"
g_inputImageDir = None
g_allInputImageFiles = None
g_allInputImageFilesReadIdx = 0
g_ldPreProcImgsDir = None
g_allNumpyInputImageFiles = None
g_allNumpyInputImageFilesReadIdx = 0
g_batchSize = 4
g_fpgaBatchSize = g_batchSize # can later become half if 8-bit
g_numDevices = 1
g_useBlas = False
g_bypassFC = False
g_bypassLoad = False
g_zmqPub = False
g_perpetual = False
g_numImages = None
g_numProcessed = 0
g_xdnnv3 = False    
g_img_c = 3
g_img_h = 224
g_img_w = 224
g_paddedImageSize = -1
g_is8bitMode = False
manager = Manager()

class PerformanceProfiler:
  def __init__(self):
    self._iteration_time = manager.dict()
    self._cumulative_time = manager.dict()
    self._iterations = manager.dict()
    self._local_iteration_time = {}
    self._local_cumulative_time = {}
    self._local_iterations = {}

    self._bars = None

  def __exit__(self, exc_type, exc_value, traceback):
    self.stop()

  def stop(self):
    if self._bars:
      for bar in self._bars:
        bar.close()
    self._bars = None

  def addSample(self, name, t):
    self._local_iteration_time[name] = 1000 * t # ms

    if name not in self._local_cumulative_time:
        self._local_cumulative_time[name] = 0.0
    else:    
        self._local_cumulative_time[name] += t
    
    if name not in self._local_iterations:
        self._local_iterations[name] = 0
    else:
        self._local_iterations[name] += 1

    # To re-enable realtime bars, we have to sync on each N updates.
    # But for now disabling realtime sync because it is expensive.
    # Each process needs to call syncToShared to update the master dict()
    #self.syncToShared()

  def syncToShared(self):
    # Each process needs to call syncToShared to push stats back to 
    # the shared master dictionaries. 
    # This is because syncing constantly is expensive.
    toSync = ["_iteration_time", "_cumulative_time", "_iterations"]
    for a in toSync:
      local = getattr(self, "_local" + a)
      shared = getattr(self, a)
      for (key, value) in local.iteritems():
        shared[key] = value

  def printSummaryRow(self, keys):
    for i,k in enumerate(keys):
      if k not in self._iterations:
        continue

      key = keys[i]
      val = 1000.*self._cumulative_time[key]/(self._iterations[key]-1)

      keysThatNeedBatchSzMult = ["imread", "resize", "meanSubtract"]
      if key in keysThatNeedBatchSzMult:
        val *= g_batchSize

      print('{:>30}'.format(key + "  |"),'{:>7}'.format('{0:.2f}'.format(val)), " ms") 

  def printSummary(self):
    self.stop()

    for i in range(14):
      print(" " * 100)

    print("===========================================")
    print("Pipeline Performance Summary\n")
    print("prep_process:")
    toPrint = ["imread", "resize", "meanSubtract", "quantizeInputs", "formatInputForXDNN"]
    self.printSummaryRow(toPrint)
    print("-------------------------------------------")
    toPrint = ["prepareImages","loadNpyImages", "putImages"]
    self.printSummaryRow(toPrint)
    
    print("===========================================")
    print("xdnn_process:")
    toPrint = ["getImages", "passThruInputsForFpga", 
      "execute (latency)", "execute (thruput)", "putFpgaOutputs"]
    self.printSummaryRow(toPrint)
    print("===========================================")
    print("post_process:")
    toPrint = ["getFpgaOutputs", "fullyConnected", "softmax", "reportAccuracy"]
    self.printSummaryRow(toPrint)
    print("===========================================\n")

  #def initBars(self):
  #  barsToCreate = [ \
  #                   "{:<30}".format("prepareImages"),
  #                   "{:<30}".format("imread"),
  #                   "{:<30}".format("resize"),
  #                   "{:<30}".format("meanSubtract"),
  #                   "{:<30}".format("quantizeInputs"),
  #                   "{:<30}".format("putImages"),
  #                   "{:<30}".format("getImages"),
  #                   "{:<30}".format("passThruInputsForFpga"),
  #                   "{:<30}".format("execute"),
  #                   "{:<30}".format("putFpgaOutputs"),
  #                   "{:<30}".format("getFpgaOutputs"),
  #                   "{:<30}".format("fullyConnected"),
  #                   "{:<30}".format("softmax"),
  #                   "{:<30}".format("Loop Time") \
  #                 ]

  #  self._bars = []
  #  for barDesc in barsToCreate:
  #    self._bars.append(tqdm(total=50, desc=barDesc, ncols=100,  
  #      leave=False, bar_format="{desc}: {bar} {n:02.2f}ms"))
 
  #def drawBars(self, batchSize, loopTime):
  #  if self._iterations["execute"] < 1:
  #    return # don't draw 1st iteration

  #  if self._bars == None:
  #    self.initBars()

  #  barVals = [ \
  #             self._iteration_time["prepareImages"],
  #             batchSize*self._iteration_time["imread"],
  #             batchSize*self._iteration_time["resize"],
  #             batchSize*self._iteration_time["meanSubtract"],
  #             self._iteration_time["quantizeInputs"],
  #             self._iteration_time["putImages"],
  #             self._iteration_time["getImages"],
  #             self._iteration_time["passThruInputsForFpga"],
  #             self._iteration_time["execute"],
  #             self._iteration_time["putFpgaOutputs"],
  #             self._iteration_time["getFpgaOutputs"],
  #             self._iteration_time["fullyConnected"],
  #             self._iteration_time["softmax"],
  #             loopTime \
  #            ]
  #  for i,bar in enumerate(self._bars):
  #    bar.n = barVals[i]
  #    bar.refresh()

g_perfProf = PerformanceProfiler()

# Timer Decorator
def timer(function):
  def functiontimer(*args, **kwargs):
    start = timeit.default_timer()
    value = function(*args,**kwargs)
    end = timeit.default_timer()
    elapsed = end - start
    
    g_perfProf.addSample(function.__name__, elapsed)
        
    return value
  return functiontimer

@timer
def imread (f):
  bgr_array = PyTurboJPEG.imread(f)
  bgr_array = cv2.resize(bgr_array,(g_img_h,g_img_w))
  return bgr_array

@timer
def resize(img):
  return cv2.resize(img,(g_img_h,g_img_w))

@timer
def meanSubtract(img,raw_scale,mean,input_scale):
  if raw_scale != 255:
      array = img.astype(np.float32) / 255
      array *= raw_scale
  array = img - mean
  if input_scale != 1:
      array *= input_scale
  return np.transpose(array,(2,0,1))

def loadImages():
  # Build List of Images We will be preparing
  global g_allInputImageFiles
  global g_allNumpyInputImageFiles

  if g_allInputImageFiles is None:
    # first time -- collect files from dir
    from os import listdir
    from os.path import isfile, join
    dirents = listdir(g_inputImageDir)
    #dirents = dirents[:4096] # ANDBG
    g_allInputImageFiles = [join(g_inputImageDir, f) \
      for f in dirents if isfile(join(g_inputImageDir, f))]
    
    #for i in range(12500):    
    #  g_allInputImageFiles.append(g_allInputImageFiles[0])
    numOrigImages = len(g_allInputImageFiles)
    numTestImages = g_batchSize
    if numOrigImages > 0 and numOrigImages < numTestImages:
      # batch size is greater than available images
      # fill up to batch size by reusing existing images
      i = 0
      while len(g_allInputImageFiles) < numTestImages:
        g_allInputImageFiles.append(g_allInputImageFiles[i])
        i = (i+1) % numOrigImages
  
  if g_ldPreProcImgsDir is not None and g_allNumpyInputImageFiles is None:
    dirents = listdir(g_ldPreProcImgsDir)
    #dirents = dirents[:4096] # ANDBG
    g_allNumpyInputImageFiles = [join(g_ldPreProcImgsDir, f) \
      for f in dirents if isfile(join(g_ldPreProcImgsDir, f))]
    
    #for i in range(12500):    
    #  g_allInputImageFiles.append(g_allInputImageFiles[0])
    numOrigImages = len(g_allNumpyInputImageFiles)
    numTestImages = g_batchSize
    #FIX ME TO_DO CHECK HOW TO BUNDLE UP
    if numOrigImages > 0 and numOrigImages < numTestImages:
      # batch size is greater than available images
      # fill up to batch size by reusing existing images
      i = 0
      while len(g_allNumpyInputImageFiles) < numTestImages:
        g_allNumpyInputImageFiles.append(g_allNumpyInputImageFiles[i])
        i = (i+1) % numOrigImages
   
@timer 
def loadNpyImages(sharedNpArr):
  global g_allNumpyInputImageFilesReadIdx 
  global g_allInputImageFilesReadIdx

  if g_allInputImageFilesReadIdx >= len(g_allInputImageFiles):
    return (None, None)
  
  inputImageFiles = []
 
  if g_allInputImageFiles:
    fname = g_allInputImageFiles[g_allInputImageFilesReadIdx].split('/')[3]+".npy"
    fname = g_ldPreProcImgsDir + "/"+fname
    inputImageFiles.append(fname)
    temp = np.load(fname)
    np.copyto(sharedNpArr, temp)
    g_allNumpyInputImageFilesReadIdx += 1
    g_allInputImageFilesReadIdx += g_batchSize
    if g_perpetual:
      g_allNumpyInputImageFilesReadIdx \
        = g_allNumpyInputImageFilesReadIdx % len(g_allNumpyInputImageFiles)
  return (sharedNpArr, inputImageFiles)

@timer
def prepareImages(sharedNpArr):
  global g_allInputImageFilesReadIdx

  if g_allInputImageFilesReadIdx >= len(g_allInputImageFiles):
    return (None, None)

  # The below needs to be parameterized
  global g_img_c
  global g_img_h
  global g_img_w

  inputs = np.empty((g_batchSize, g_img_c, g_img_h, g_img_w), dtype=np.float32)
  inputImageFiles = []
  img_num = 0

  if g_allInputImageFiles:
      # use raw image files from user
    while img_num < g_batchSize \
      and g_allInputImageFilesReadIdx < len(g_allInputImageFiles):
      fname = g_allInputImageFiles[g_allInputImageFilesReadIdx]
      inputImageFiles.append(fname)
      if not g_bypassLoad:
        temp = imread(fname)
        temp = resize(temp)
        inputs[img_num] = meanSubtract(temp,g_raw_scale,g_mean,g_input_scale)
        #inputs[img_num] \
        #  = xdnn_io.loadImageBlobFromFile(fname, g_raw_scale, g_mean, g_input_scale, g_img_h, g_img_w)
      img_num += 1
      g_allInputImageFilesReadIdx += 1
      if g_perpetual:
        g_allInputImageFilesReadIdx \
          = g_allInputImageFilesReadIdx % len(g_allInputImageFiles)
    
    if not g_bypassLoad:
      fpgaInputs = quantizeInputs(g_firstFpgaLayerName, g_fpgaCfgFile, g_scaleB, inputs)
      if g_xdnnv3:
        fpgaInputs = formatInputBatchNpyForXDNN(fpgaInputs, inputImageFiles)
#        fpgaInputs = formatInputForXDNN(fpgaInputs)
      np.copyto(sharedNpArr, fpgaInputs.flatten())
  return (sharedNpArr, inputImageFiles)
  
@timer
def quantizeInputs(g_firstFpgaLayerName,g_fpgaCfgFile,g_scaleB,inputs):
  return xdnn.quantizeInputs(g_firstFpgaLayerName, g_fpgaCfgFile, g_scaleB, inputs)

@timer
def putImages(item,q):
  q.put(item)

@timer
def getImages(q):
  return q.get()

@timer
def putFpgaOutputs(item,q):
  q.put(item)

@timer
def getFpgaOutputs(q):
  return q.get()

@timer
def fullyConnected(fcWeight,fcBias,fpgaOutput,g_batchSize,g_outputSize,g_fpgaOutputSize,g_useBlas):
  return xdnn.computeFC(fcWeight,fcBias,fpgaOutput,g_batchSize,g_outputSize,g_fpgaOutputSize,g_useBlas)

@timer
def softmax(fcOutput,g_batchSize):
  return xdnn.computeSoftmax(fcOutput,g_batchSize)

def v3PaddedImgSize(inRows, inCols, inChans):
  if g_is8bitMode:
    bytesPixel = 1
  else:
    bytesPixel = 2
  ddrWordBytes = 64
  pixelGroupSize = 8
  ddrWordElementsNum = ddrWordBytes/bytesPixel
  ddrWordPixelGroups = ddrWordElementsNum / pixelGroupSize
  numDdrWords = int(math.ceil(float(inRows)/float(ddrWordPixelGroups)))
  xdnnv3ImgSize = inCols*numDdrWords*32*int(math.ceil(float(inChans)/float(pixelGroupSize)))
  if g_is8bitMode:
    xdnnv3ImgSize = xdnnv3ImgSize * 2 #size in char's
  return xdnnv3ImgSize
 
def processCommandLine():
  global g_xclbin
  global g_netFile
  global g_fpgaCfgFile
  global g_xdnnTestDataDir
  global g_labels
  global g_goldenFile
  global g_inputImageDir
  global g_ldPreProcImgsDir
  global g_xdnnLib
  global g_doQuant
  global g_fpgaOutputSize
  global g_outputSize
  global g_firstFpgaLayerName
  global g_useBlas
  global g_bypassLoad
  global g_bypassFC
  global g_zmqPub
  global g_perpetual
  global g_numImages
  global g_batchSize
  global g_fpgaBatchSize
  global g_xdnnv3
  global g_paddedImageSize
  global g_is8bitMode

  parser = argparse.ArgumentParser(description='pyXDNN')
  parser.add_argument('--usexdnnv3', action='store_true',
    help='version of xdnn')
  parser.add_argument('--xclbin',
    help='.xclbin file')
  parser.add_argument('--netcfg',
    help='FPGA instructions generated by compiler for the network')
  parser.add_argument('--quantizecfg',
    help='FPGA config file')
  parser.add_argument('--xlnxlib',
    help='FPGA xfDNN lib .so')
  parser.add_argument('--fpgaoutsz',
    help='size of 1 FPGA output blob')
  parser.add_argument('--outsz',
    help='size of 1 output blob')
  parser.add_argument('--firstfpgalayer',
    help='name of first FPGA layer (to start quantization)')
  parser.add_argument('--datadir',
    help='path to data files to run for the network')
  parser.add_argument('--labels',
    help='result -> labels translation file')
  parser.add_argument('--golden',
    help='file idx -> expected label file')
  parser.add_argument('--imagedir',
    help='directory with image files to classify')
  parser.add_argument('--loadPreProcessedImagesDir',
    help='directory with pre processed numpy image files to classify')
  parser.add_argument('--useblas', action='store_true',
    help='use BLAS-optimized functions (requires xfDNN lib compiled with BLAS)')
  parser.add_argument('--zmqpub', 
    help='publish predictions to zmq port 5555')
  parser.add_argument('--perpetual', action='store_true', 
    help='loop over input images forever')
  parser.add_argument('--bypassFC', action='store_true',
    help='This flag will skip fully connected layer')
  parser.add_argument('--bypassLoad', action='store_true',
    help='This flag will skip loading real images')
  parser.add_argument('--numImages', 
    help='Break when numImages have been processed')
  parser.add_argument('--batchSize', 
    help='Images to process in parallel')
  args = parser.parse_args()

  if os.path.isfile(args.xclbin) and os.access(args.xclbin, os.R_OK):
    g_xclbin = args.xclbin
  else:
    sys.exit("ERROR: Specified xclbin file does not exist or is not readable.")

  if os.path.isfile(args.netcfg) and os.access(args.netcfg, os.R_OK):
    g_netFile = args.netcfg
  else:
    sys.exit("ERROR: Specified netcfg file does not exist or is not readable.")

  if os.path.isfile(args.xlnxlib) and os.access(args.xlnxlib, os.R_OK):
    g_xdnnLib = args.xlnxlib
  else:
    sys.exit("ERROR: Specified xlnxlib file does not exist or is not readable.")
  if args.usexdnnv3:
    g_xdnnv3 = True

  if args.fpgaoutsz:
    g_fpgaOutputSize = int(args.fpgaoutsz)
  if args.outsz:
    g_outputSize = int(args.outsz)
  if args.firstfpgalayer:
    g_firstFpgaLayerName = args.firstfpgalayer

  if os.path.isdir(args.datadir) and os.access(args.datadir, os.R_OK):
    g_xdnnTestDataDir = args.datadir
  else:
    sys.exit("ERROR: Specified datadir directory does not exist or is not readable.")

  if os.path.isfile(args.labels) and os.access(args.labels, os.R_OK):
    g_labels = []
    with open(args.labels, 'r') as f:
      for line in f:
        g_labels.append(line.strip())
  else:
    sys.exit("ERROR: Specified labels file does not exist or is not readable.")

  if os.path.isfile(args.golden) and os.access(args.golden, os.R_OK):
    g_goldenFile = args.golden
  else:
    sys.exit("ERROR: Specified golden file does not exist or is not readable.")

  if os.path.isdir(args.imagedir) and os.access(args.imagedir, os.R_OK):
    g_inputImageDir = args.imagedir
  else:
    sys.exit("ERROR: Specified imagedir directory %s does not exist or is not readable." % args.imagedir)

  if args.loadPreProcessedImagesDir:
    if os.path.isdir(args.loadPreProcessedImagesDir) and os.access(args.loadPreProcessedImagesDir, os.R_OK):
      g_ldPreProcImgsDir = args.loadPreProcessedImagesDir
    else:
      sys.exit("ERROR: Specified pre processed numpy imagedir directory %s does not exist or is not readable." % args.g_ldPreProcImgsDir)

  if args.useblas:
    g_useBlas = True
  if args.bypassLoad:
    g_bypassLoad = True
  if args.bypassFC:
    g_bypassFC = True
  if args.zmqpub:
    g_zmqPub = True
  if args.perpetual:
    print("Running Perpetually")
    g_perpetual = True
  if args.numImages:
    g_numImages = int(args.numImages)
  if args.batchSize:
    g_batchSize = int(args.batchSize)
    g_fpgaBatchSize = g_batchSize

  if args.quantizecfg:
    if os.path.isfile(args.quantizecfg):
      g_fpgaCfgFile = args.quantizecfg
      g_doQuant = True
      with open(g_fpgaCfgFile) as qfile:    
        qObj = json.load(qfile)
        if int(qObj['network'][0]['bw_params']) == 8:
          g_is8bitMode = True
          if not g_xdnnv3:
            # 8-bit inputs are packed
            g_fpgaBatchSize = int(math.ceil(float(g_batchSize) / 2.))
  else:
    g_fpgaCfgFile = ""
  g_paddedImageSize = g_img_c*g_img_h*g_img_w
  # TODO FIXME STOPPED HERE
  # if xdnn v3, use function to compute padded image size 
  # check also if 'is8bitMode'
  if g_xdnnv3:
    g_paddedImageSize = v3PaddedImgSize(g_img_w, g_img_h, g_img_c)

def std2xdnnv3(srcImg):
  srcImg = srcImg.flatten()
  if g_is8bitMode == True:
    bytesPixel = 1
  else:
    bytesPixel = 2
  pixelGroupSize = 8
  ddrWordBytes = 64
  
  numChnlGroup = int(math.ceil(float(g_img_c)/float(pixelGroupSize)))
  destIdx = 0
  srcIdx = 0
  dest=np.ndarray((g_paddedImageSize))
  for i in range(numChnlGroup):
    for j in range(g_img_w):
      byteCount = 0
      for k in range(g_img_h):
        for d in range(pixelGroupSize):
          byteCount = byteCount + bytesPixel
          chnlIdx = i*pixelGroupSize + d
          if chnlIdx<g_img_c:
            srcIdx = (chnlIdx*g_img_w*g_img_h) +\
               (j*g_img_h)+k
            dest[destIdx] = srcImg[srcIdx]
            
            destIdx = destIdx + 1
          else:
            dest[destIdx] = 0
            destIdx = destIdx + 1
      while byteCount%ddrWordBytes!=0:
        dest[destIdx] = 0
        destIdx = destIdx + 1
        byteCount = byteCount + bytesPixel
  return dest

@timer
def formatInputForXDNN(src):
  dest = np.ndarray(shape=(src.shape[0],g_paddedImageSize), dtype = int)
  for i in range(src.shape[0]):
    dest[i] = std2xdnnv3(src[i])
  return dest

def formatInputNpyForXDNN(src, inputImageFiles):
  dest = np.ndarray(shape=(src.shape[0],g_paddedImageSize), dtype = np.int8)
  for i in range(len(inputImageFiles)):
    inputImageFiles[i]=inputImageFiles[i].split('/')[3]
  for i in range(src.shape[0]):
    dest[i] = std2xdnnv3(src[i])
    fpgaInputs = dest[i].flatten()
    np.save('/scratch/ilsvrc12_img_val_npyint8/'+inputImageFiles[i]+'.npy', fpgaInputs)
  return dest

def formatInputBatchNpyForXDNN(src, inputImageFiles):
  dest = np.ndarray(shape=(src.shape[0],g_paddedImageSize), dtype = np.int8)
  for i in range(len(inputImageFiles)):
    inputImageFiles[i]=inputImageFiles[i].split('/')[3]
  for i in range(src.shape[0]):
    dest[i] = std2xdnnv3(src[i])
  np.save('/scratch/ilsvrc12_img_val_realnpyint8batch/'+inputImageFiles[0]+'.npy', dest.flatten())
 
  return dest


##############################################################################
## Start prep_process ########################################################
##############################################################################
def prep_process(q, sharedInputArrs):
  global g_numImages
  global g_numProcessed 

  #p_history = {}
  #p_history["y"] = []
  #p_history["t"]  = []

  ret = xdnn.createManager(g_xdnnLib)
  if ret != True:
    sys.exit(1)

  shMemIdx = -1
  while True:
    #p_history["y"].append(1)
    #p_history["t"].append(timeit.default_timer())
    shMemIdx = (shMemIdx + 1) % len(sharedInputArrs)
    # WARNING: shared mem below is not synchronized.
    # currently relies on shared mem banks to be consumed faster 
    # than the next cycle of writes can come long.
    # Be sure to add enough shared mem banks to feed FPGA.
    sharedInputArr = sharedInputArrs[shMemIdx]
    sharedNpArr = np.frombuffer(sharedInputArr, np.int16)

    if g_ldPreProcImgsDir is not None:
      (fpgaInputs, inputImageFiles) = loadNpyImages(sharedNpArr)
    else:
      (fpgaInputs, inputImageFiles) = prepareImages(sharedNpArr)

    
    if fpgaInputs is None:
      break
    putImages(shMemIdx, q)

    #p_history["y"].append(0)
    #p_history["t"].append(timeit.default_timer())

  #plt.plot(np.array(p_history["t"]),np.array(p_history["y"]))
  #plt.show()

  #print p_history

  q.put(None)
  g_perfProf.syncToShared()

##############################################################################
## Start xdnn_process ########################################################
##############################################################################
def xdnn_process (qFrom, qTo, qMsgFromXdnn, sharedInputArrs):
  
  global g_numImages
  global g_numProcessed 

  global g_img_c
  global g_img_h
  global g_img_w

  xdnn_handle = xdnn.createHandle(g_xclbin, "kernelSxdnn_0", g_xdnnLib, g_numDevices)
  if xdnn_handle != 0:
    sys.exit(1)      

  fpgaOutputs = []
  for inp in sharedInputArrs:
    fpgaOutputs.append(xdnn_io.prepareOutput(g_fpgaOutputSize, g_batchSize))

  # load weights
  args = { 'datadir': g_xdnnTestDataDir, 
           'quantizecfg': g_fpgaCfgFile, 
           'scaleA': g_scaleA, 
           'scaleB': g_scaleB,
           'PE': -1,
           'netcfg': g_netFile }
  if g_xdnnv3 == True:
    weightsBlob = xdnn_io.loadWeightsBiasQuantv3(args)
  else:
    weightsBlob = xdnn_io.loadWeightsBiasQuant(args)
  
  # Dummy calls to load script
  for streamId in range(len(sharedInputArrs)):
    fpgaInputs = xdnn.passThruInputsForFpga(sharedInputArrs[streamId],
      g_fpgaBatchSize, g_paddedImageSize,
      g_fpgaCfgFile,g_scaleB,-1,g_firstFpgaLayerName, streamId)
    xdnn.exec_async(g_netFile,weightsBlob,fpgaInputs,fpgaOutputs[streamId], 
      g_batchSize,g_fpgaCfgFile,g_scaleB, -1, streamId)
    xdnn.get_result(-1, streamId)

  # XDNN Is Ready to Rock
  qMsgFromXdnn.put(timeit.default_timer()) # Share Start Time
  
  print("Streaming...")
  pendingJobQ = []
  while True:
    streamId = getImages(qFrom)
    if streamId is None:
      # finish pending jobs & quit
      for (streamId, startTime) in pendingJobQ:
        xdnn.get_result(-1, streamId)
        now = timeit.default_timer()
        g_perfProf.addSample("execute (latency)", now-startTime)
        g_perfProf.addSample("execute (thruput)", now-startTime)

        putFpgaOutputs(fpgaOutputs[streamId], qTo)
      break

    startTime = timeit.default_timer()
    fpgaInputs = xdnn.passThruInputsForFpga(sharedInputArrs[streamId], 
      g_fpgaBatchSize, g_paddedImageSize,
      g_fpgaCfgFile, g_scaleB, -1, g_firstFpgaLayerName, streamId)
    g_perfProf.addSample("passThruInputsForFpga", timeit.default_timer()-startTime)
    if not fpgaInputs:
      break
    
    startTime = timeit.default_timer()
    xdnn.exec_async(g_netFile, weightsBlob, fpgaInputs, fpgaOutputs[streamId],
      g_batchSize, g_fpgaCfgFile, g_scaleB, -1, streamId)
    pendingJobQ.append((streamId, startTime))

    if len(pendingJobQ) >= len(fpgaOutputs):
      # pop oldest job off the q and get_result
      (streamId, jobStartTime) = pendingJobQ.pop(0)      
      xdnn.get_result(-1, streamId)
      now = timeit.default_timer()
      g_perfProf.addSample("execute (latency)", now-jobStartTime)
      g_perfProf.addSample("execute (thruput)", now-startTime)

      putFpgaOutputs(fpgaOutputs[streamId], qTo)
  
  qTo.put (None)
  g_perfProf.syncToShared()
  xdnn.closeHandle()        

class ZmqResultPublisher:
  def __init__(self):
    import zmq
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.socket.bind("tcp://*:5555")

  def send(self, data):
    self.socket.send(data)
        
##############################################################################
## Start post_process ########################################################
##############################################################################
def post_process():
  global g_numProcessed 
  processCommandLine()
  ret = xdnn.createManager(g_xdnnLib)
  if ret != True:
    sys.exit(1)

  loadImages()
  (fcWeight, fcBias) = xdnn_io.loadFCWeightsBias(g_xdnnTestDataDir)

  # sharedInputArrs = rolling bank of shared memory blocks
  # -- 1 bank for each stream
  sharedInputArrs = []
  for i in range(4):
    sharedInputArrs.append(
      sharedctypes.RawArray(\
      ctypes.c_short, g_fpgaBatchSize*g_paddedImageSize))

  # Spawn the first 2 stages of our pipeline
  # Stage 1: Process JPG
  # Stage 2: Run FPGA "classify"
  qFpga = Queue(maxsize=1)
  qPrep = Queue(maxsize=1)
  qMsgFromXdnn = Queue(maxsize=1)

  # start FPGA proc first to make sure FPGA is done initializing
  xdnnProc = Process(target=xdnn_process, 
    args=(qPrep, qFpga, qMsgFromXdnn, sharedInputArrs)) 
  xdnnProc.start() 

  # only start prep proc after FPGA xdnn proc is ready
  xdnnReady = qMsgFromXdnn.get()
  prepProc = Process(target=prep_process,
    args=(qPrep, sharedInputArrs))  
  prepProc.start() 

  #
  # The rest of this function post-processes FPGA output:
  # 1) Compute the final FC + Softmax layers
  # 2) Print classification & accuracy
  # 
  zmqPub = None
  if g_zmqPub:
    zmqPub = ZmqResultPublisher()
  goldenMap = None
  if g_goldenFile:
    goldenMap = getGoldenMap(g_goldenFile)
  g_numProcessed = 0
  allTop1 = 0
  allTop5 = 0
  
  startTime = None
  while True:    
    loopTime = timeit.default_timer()*(-1)
    fpgaOutput = getFpgaOutputs(qFpga)

    if g_numImages is not None and g_numProcessed >= g_numImages:
      break

    if type(fpgaOutput) == type(None):
      break

    inputImageFiles = []
    for i in range(g_batchSize):
      idx = (g_numProcessed + i) % len(g_allInputImageFiles)
      inputImageFiles.append(g_allInputImageFiles[idx])

    if g_bypassFC:
      fcOutput = np.zeros(g_batchSize*g_outputSize)
    else:
      fcOutput = fullyConnected(fcWeight,fcBias,fpgaOutput,g_batchSize,g_outputSize,g_fpgaOutputSize,g_useBlas)
    smaxOutput = softmax(fcOutput,g_batchSize)
    loopTime += timeit.default_timer()
    loopTime *= 1000 # ms 
    g_numProcessed += g_batchSize

    if not g_bypassLoad:
      (top1, top5) = reportAccuracy(smaxOutput.flatten().tolist(), 
        g_outputSize, inputImageFiles, g_labels, goldenMap,zmqPub,True)
      allTop1 += top1
      allTop5 += top5
    
    #g_perfProf.drawBars(g_batchSize, loopTime)

    if startTime == None:
      # set startTime after skipping 1st iteration
      startTime = timeit.default_timer()

  endTime = timeit.default_timer()
  elapsed = endTime - startTime
  elapsed *= 1000

  prepProc.join()
  xdnnProc.join()

  g_perfProf.syncToShared()
  g_perfProf.printSummary()
  
  if g_numProcessed > 1:
    numProfiled = g_numProcessed - 1 # we skipped 1 iter to flush pipe
    print("===========================================")
    print("Performance Summary\n")
    print("  Images: %d" % (g_numProcessed)) 
    if goldenMap is not None:
      print("  Top1: %.2f%%" % (100*allTop1/float(g_numProcessed))) 
      print("  Top5: %.2f%%" % (100*allTop5/float(g_numProcessed))) 
    print("  Batch Size: %d" % (g_batchSize)) 
    print("  Total Batches: %d" % (numProfiled/g_batchSize)) 
    print("  Total Time: %.2f ms" % (elapsed))
    print("  Time/Batch: %.2f ms" % (g_batchSize*elapsed/numProfiled))
    print("  Time/Image: %.2f ms" % (elapsed/numProfiled))
    print("  Images/Second: %f" % (1000*numProfiled/elapsed))
    print("===========================================\n")

def getGoldenMap(goldenFile):
  #goldenMap = collections.OrderedDict()
  goldenMap = {}
  with open(goldenFile, 'r') as f:
    for line in f:
      fname = line[:line.rfind(' ')]
      goldenIdx = int(line[line.rfind(' ')+1:])
      goldenMap[fname] = goldenIdx

  return goldenMap

def checkAccuracy(fname, result, goldenMap):
  # get idx from file name ILSVRC2012_val_00031091.JPEG
  goldenIdx = goldenMap[fname]

  # result is in the form [(value, idx), (value, idx), ...]
  for i,(val,idx) in enumerate(result):
    if i == 0 and idx == goldenIdx:
      return (1, 1)
    elif idx == goldenIdx:
      return (0, 1)

  return (0, 0)

@timer
def reportAccuracy(output, outputSize, inputImageFiles, labels, goldenMap=None, zmqPub=None, silent = True):
  top1count = 0
  top5count = 0

  silent or print("\n")
  zmqMessage = ""
  for i in range(len(inputImageFiles)):
    silent or print ("---------- Prediction %d for %s ----------" \
      % (i, inputImageFiles[i]))
    if zmqPub:
      zmqMessage += "%s\n" % inputImageFiles[i]

    startIdx = i * outputSize
    vals = output[startIdx:startIdx + outputSize]
    top5Idx = np.argsort(vals)[-5:]
    top5Vals = [vals[ti] for ti in top5Idx]
    top5 = []
    for ti, _ in enumerate(top5Idx):
      top5.append((top5Vals[ti], top5Idx[ti]))
    top5.reverse()

    for j in range(len(top5)):
      confidence = top5[j][0]
      label = labels[top5[j][1]]
      silent or print ("%.4f - \"%s\"" % (confidence, label))
      if zmqPub:
        zmqMessage += "%.4f %s\n" % (confidence, label)

    if goldenMap != None:
      (t1, t5) = checkAccuracy(os.path.split(inputImageFiles[i])[1], 
        top5, goldenMap)
      top1count += t1
      top5count += t5
      silent or print("Accuracy (i=%d) Top-1: %d, Top-5: %d" \
        % (g_numProcessed/g_batchSize, top1count, top5count))

  if zmqPub:
    zmqPub.send(zmqMessage)

  return (top1count, top5count)


if __name__ == '__main__':
  post_process()

