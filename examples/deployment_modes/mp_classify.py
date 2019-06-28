#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import sys
import timeit
import numpy as np
import multiprocessing as mp
import ctypes
import signal
import threading
from xfdnn.rt import xdnn, xdnn_io
import time


###################################################
# Pre-process
###################################################
class UserPreProcess():
  def __init__(self, args, q, img_paths, sharedInputArrs, prepProcQ, input_shapes):
    if xdnn.createManager() != True:
      sys.exit(1)
    np.random.seed(123)  # for reproducibility
    self._args = args
    self._firstInputShape = input_shapes[0]
    self._q = q
    self._imgpaths = img_paths
    current = mp.current_process()
    self._procid = (int(current._identity[0]) - 1) % args['numprepproc']
    self._sharedmem = sharedInputArrs
    self._prepQ = prepProcQ

    #HWC format as this is the native format that comes out of jpeg decode
    self._meanarr = np.zeros ( (self._firstInputShape[2], self._firstInputShape[3], self._firstInputShape[1],), dtype = np.float32, order='C' )
    self._meanarr += args['img_mean']

  def run(self, inum):
    buf_id = self._prepQ.get()

    if not self._args['benchmarkmode']:
      np_arr = np.frombuffer(self._sharedmem[buf_id][0].get_obj(), dtype = np.float32)
      np_arr = np.reshape ( np_arr, (1,) + tuple(self._firstInputShape[1:]), order = 'C')
      np_arr[:], _ = xdnn_io.loadImageBlobFromFile(self._imgpaths[inum], self._args['img_raw_scale'], self._meanarr,
                                             self._args['img_input_scale'], self._firstInputShape[2], self._firstInputShape[3])
    self._q.put ( (buf_id, inum, None) )

###################################################
# Post-process
###################################################

class ZmqResultPublisher:
  def __init__(self, devid):
    import zmq
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.socket.bind("tcp://*:55{}5".format(devid))

  def send(self, data):
    self.socket.send(data)

class UserPostProcess():
  def __init__(self, qFrom, args, img_paths, streamQ, fpgaOutputs, output_shapes):
    (self.fcWeight, self.fcBias) = xdnn_io.loadFCWeightsBias(args)
    self.qFrom = qFrom
    self.args = args
    self.img_paths = img_paths
    self.streamQ = streamQ
    self.fpgaOutputs = fpgaOutputs
    self.output_shapes = output_shapes

    self.numProcessed = 0
    self.startTime = timeit.default_timer()

  #
  # This function post-processes FPGA output:
  # 1) Compute the final FC + Softmax layers
  # 2) Print classification & accuracy
  #
  def run(self, sId, imgList, fpgaOutput_list, fpgaOutputShape_list):
    fpgaOutput  = fpgaOutput_list[0]
    fpgaOutputShape  = fpgaOutputShape_list[0]
    if self.numProcessed == 0:
      self.startTime = timeit.default_timer()
      self.labels = xdnn_io.get_labels(self.args['labels'])
      self.zmqPub = None
      if self.args['zmqpub']:
        self.zmqPub = ZmqResultPublisher(self.args['deviceID'])
      self.goldenMap = None
      if self.args['golden']:
        self.goldenMap = xdnn_io.getGoldenMap(self.args['golden'])
        self.top5Count = 0
        self.top1Count = 0
      self.fcOutput = np.empty((self.args['batch_sz'], self.args['outsz'],), 
        dtype=np.float32, order='C')

    self.numProcessed += len(imgList)
  
    npout_view = np.frombuffer(fpgaOutput, dtype=np.float32)\
      .reshape(tuple(fpgaOutputShape))
    xdnn.computeFC(self.fcWeight, self.fcBias, npout_view, self.fcOutput)

    self.streamQ.put(sId)
  
    smaxOutput = xdnn.computeSoftmax(self.fcOutput)
    if self.args['golden']:
      for i,p in enumerate ( imgList ):
        self.top1Count += xdnn_io.isTopK(\
          smaxOutput[i], self.goldenMap, p, self.labels, 1)
        self.top5Count += xdnn_io.isTopK(\
          smaxOutput[i], self.goldenMap, p, self.labels, 5)
  
    if self.zmqPub is not None:
      predictMsg = xdnn_io.getClassification(\
        smaxOutput, imgList, self.labels, zmqPub=True)
      self.zmqPub.send(predictMsg)
  
  def loop(self):
    fpgaOutputShape = self.output_shapes[0]
    fpgaOutputShape[0] = self.args['batch_sz']

    while True:
      (sId, img_idx, userData) = self.qFrom.get()
      if sId is None or img_idx is None:
        break
  
      imgList = []
      for x in np.nditer(img_idx):
        if x >= 0:
          imgList.append(self.img_paths[x])
  
      if self.args["benchmarkmode"]:
        self.numProcessed += len(imgList)
        self.streamQ.put(sId)
        continue

      self.run(sId, imgList, [self.fpgaOutputs[sId][0].get_obj()], [fpgaOutputShape])  

    self.finish()

  def finish(self):
    print ( "%g images/s" % ( float(self.numProcessed) / (timeit.default_timer() - self.startTime )  ))
  
    if self.args['golden'] and self.numProcessed:
      print ("\nAverage accuracy (n=%d) Top-1: %.1f%%, Top-5: %.1f%%\n") \
        % (self.numProcessed,
           float(self.top1Count)/float(self.numProcessed)*100.,
           float(self.top5Count)/float(self.numProcessed)*100.)

###################################################
# Instantiate pre/post processes,
# allow user to register own classes
###################################################

g_preClass = UserPreProcess
g_postClass = UserPostProcess
g_preInst = None
g_postInst = None

def register_pre(preClass):
  global g_preClass
  g_preClass = preClass

def register_post(postClass):
  global g_postClass
  g_postClass = postClass

def init_pre_process(args, q, img_paths, sharedInputArrs, prepProcQ, input_shapes):
  global g_preClass
  global g_preInst
  g_preInst = g_preClass(args, q, img_paths, sharedInputArrs, prepProcQ, input_shapes)

def run_pre_process(imgpath_idx):
  global g_preInst
  return g_preInst.run(imgpath_idx)

def post_process(qFrom, args, img_paths, streamQ, fpgaOutputs, output_shapes):
  global g_postClass
  global g_postInst
  g_postInst = g_postClass(qFrom, args, img_paths, streamQ, fpgaOutputs, output_shapes)
  g_postInst.loop()

###################################################
# FPGA
###################################################

def fpga_wait( fpgaRT, q, qPost, prepProcQ):
  numProcessed = 0
  while True:
    i,img_list, sMemListIdx, userData  = q.get()
    if i is None:
      break

    numProcessed += len(img_list)
    fpgaRT.get_result(i)
    for j in sMemListIdx:
      prepProcQ.put(j)
    qPost.put ( (i, img_list, userData) )

  qPost.put ( ( None, None, None ))

def fpga_process(fpgaRT, qFrom, qTo, args, num_img, sharedInputArrs, prepProcQ,  streamQ, fpgaOutputs, compJson):
    if fpgaRT is None:
        ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", [args["deviceID"]])
        if ret != 0:
            sys.exit(1)
        fpgaRT = xdnn.XDNNFPGAOp(handles, args)
    else:
        print "fpga process handle was ready:"
    qWait = mp.Queue(maxsize=100)
    numStreams = args['numstream']
    bsz = args['batch_sz']
    input_ptrs = [[] for i in range(numStreams)]
    
    
    numProcessed = 0
    t = threading.Thread(target=fpga_wait, args=(fpgaRT, qWait, qTo, prepProcQ, ))
    t.start()
    
    input_shapes = map(lambda x: (x), compJson.getInputs().itervalues())
    output_shapes = map(lambda x: (x), compJson.getOutputs().itervalues()) 
    
 
    InputName_list = map(lambda x: str(x), compJson.getInputs().iterkeys())
    OutputName_list = map(lambda x: str(x), compJson.getOutputs().iterkeys())
    num_inputs = len(input_shapes)
    num_outputs = len(output_shapes)

    startTime = time.time()
    while numProcessed < num_img or args['perpetual']:

        img_list = np.full( (bsz,), -1, dtype = np.int32 )
        sId = streamQ.get()
        input_ptrs[sId] = []
        shMemIdxArr = []
        userDataArr = []
        input_arrays = []
        for in_idx in range(len(input_shapes)):            
            input_arrays = []
            input_ptrs[sId].append(input_arrays)
            
        for j in range(bsz):
            (sMemIdx, img_idx, userData) = qFrom.get()
            numProcessed += 1
            img_list[j] = img_idx
            nparr_view = []
            for in_idx in range(len(input_shapes)):
                nparr_view = (np.frombuffer(sharedInputArrs[sMemIdx][in_idx].get_obj(), dtype = np.float32))
                input_ptrs[sId][in_idx].append( nparr_view )
                
            #nparr_view = np.frombuffer(sharedInputArrs[sMemIdx].get_obj(), dtype = np.float32).reshape ( tuple ( firstInputShape ))
            
            shMemIdxArr.append(sMemIdx)
            userDataArr.append(userData)
            if numProcessed == num_img:
                break
        npout_view = []
        for out_idx in range(num_outputs):
            npout_view.append(np.frombuffer(fpgaOutputs[sId][out_idx].get_obj(), dtype = np.float32).reshape( (args['batch_sz'],) + tuple ( output_shapes[out_idx][1:]) ))
        
        in_dict = {}
        out_dict = {}
        
        for in_idx in range(num_inputs):
           in_dict[InputName_list[in_idx]] = input_ptrs[sId][in_idx]
           
        for out_idx in range(num_outputs):
            out_dict[OutputName_list[out_idx]] = npout_view[out_idx]
            
           
        fpgaRT.exec_async( in_dict, out_dict, sId)

        qWait.put((sId, img_list, shMemIdxArr, userDataArr))

    qWait.put ((None, None, None, None))
    t.join()
    elapsedTime = ( time.time() - startTime )
    print ( "FPGA_process: ", float(numProcessed)/elapsedTime, "img/s")

    xdnn.closeHandle()

###################################################
# "Main"
###################################################

def run(args=None):
  if not args:
    parser = xdnn_io.default_parser_args()
    parser.add_argument('--numprepproc', type=int, default=1,
                        help='number of parallel processes used to decode and quantize images')
    parser.add_argument('--numstream', type=int, default=16,
                        help='number of FPGA streams')
    parser.add_argument('--deviceID', type=int, default=0,
                        help='FPGA no. -> FPGA ID to run in case multiple FPGAs')
    parser.add_argument('--benchmarkmode', type=int, default=0,
                        help='bypass pre/post processing for benchmarking')
    args = parser.parse_args()
    args = xdnn_io.make_dict_args(args)

  if not xdnn.createManager():
    sys.exit(1)
  fpgaRT = None

  sharedInputArrs = []
  fpgaOutputs = []
  qPrep = mp.Queue(maxsize=args['numprepproc']*10)
  qFpga = mp.Queue(maxsize=100)
  streamQ = mp.Queue(maxsize=args['numstream'])
  prepProcQ = mp.Queue(maxsize=100)
  
  
  compilerJSONObj = xdnn.CompilerJsonParser(args['netcfg'])

  input_shapes = map(lambda x: (x), compilerJSONObj.getInputs().itervalues())
  output_shapes = map(lambda x: (x), compilerJSONObj.getOutputs().itervalues())    

  input_sizes = map(lambda x: np.prod(x), input_shapes)
  output_sizes = map(lambda x: np.prod(x), output_shapes)
  
  
  for i in range( args['numstream'] ):
    fpga_output_list = []  
    for j in range(len(output_sizes)):  
        fpga_output_list.append(mp.Array(ctypes.c_float, args['batch_sz'] * output_sizes[j]) ) 
    fpgaOutputs.append(fpga_output_list)
    streamQ.put ( i )

  for i in range(100):
    fpga_in_list = []    
    for j in range(len(input_sizes)):
        fpga_in_list.append(mp.Array(ctypes.c_float, input_sizes[j] ))        
    sharedInputArrs.append( fpga_in_list )
    prepProcQ.put (i)

  img_paths = xdnn_io.getFilePaths(args['images'])
  p = mp.Pool(initializer = init_pre_process, 
    initargs = (args, qPrep, img_paths, sharedInputArrs, prepProcQ, input_shapes, ), processes = args['numprepproc'])

  xdnnProc = mp.Process(target=fpga_process, args=(fpgaRT, qPrep, qFpga, args, len(img_paths), sharedInputArrs,prepProcQ, streamQ, fpgaOutputs, compilerJSONObj,))

  postProc = mp.Process(target=post_process, args=(qFpga, args, img_paths, streamQ, fpgaOutputs,output_shapes,))

  if args['perpetual']:
    while True:
      res = [p.map_async(run_pre_process, range(len(img_paths)))]
      for j in res:
        j.wait()
        del j
  else:
    p.map_async(run_pre_process, range(len(img_paths)))

  xdnnProc.start()
  postProc.start()
  xdnnProc.join()
  postProc.join()

  p.close()
  p.join()
  
if __name__ == '__main__':
  run()
