#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import sys
import timeit
import xdnn, xdnn_io
import numpy as np
import multiprocessing as mp
import ctypes
import threading
import time
import os
manager = mp.Manager()

class ZmqResultPublisher:
  def __init__(self, devid):
    import zmq
    self.context = zmq.Context()
    self.socket = self.context.socket(zmq.PUB)
    self.socket.bind("tcp://*:55{}5".format(devid))

  def send(self, data):
    self.socket.send(data)

def xdnn_wait( fpgaRT, q, qPost, prepProcQ):
  numProcessed = 0
  while True:
    i,img_list, sMemListIdx  = q.get()
    if i is None:
      break
    
    numProcessed += len(img_list)
    fpgaRT.get_result(i)
    for j in sMemListIdx:
      prepProcQ.put(j)
    qPost.put ( (i, img_list) )
  
  qPost.put ( ( None, None ))
  
def fpga_process_async (qFrom, qTo, args, num_img, sharedInputArrs, prepProcQ,  streamQ, fpgaOutputs):
  
  ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", [args["deviceID"]])
  if ret != 0:
    sys.exit(1)
  fpgaRT = xdnn.XDNNFPGAOp(handles, args)

  qWait = mp.Queue(maxsize=100)

  numStreams = args['numstream']
  bsz = args['batch_sz']
  input_ptrs = [] 
  for i in range(numStreams):
    input_ptrs.append([])

  numProcessed = 0
  t = threading.Thread(target=xdnn_wait, args=(fpgaRT, qWait, qTo, prepProcQ, ))
  t.start()
  #startTime = time.time()
  while numProcessed < num_img or args['perpetual']:
    img_list = np.full( (bsz,), -1, dtype = np.int32 )
    sId = streamQ.get()
    input_ptrs[sId] = []
    shMemIdxArr = []
    for j in range(bsz):
      (sMemIdx, img_idx) = qFrom.get()
      numProcessed += 1 
      img_list[j] = img_idx
      nparr_view = np.frombuffer(sharedInputArrs[sMemIdx].get_obj(), dtype = np.float32)
      nparr_view = nparr_view[np.newaxis, ...]
      input_ptrs[sId].append( nparr_view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) )
      shMemIdxArr.append(sMemIdx)
      if numProcessed == num_img:
        break        

    npout_view = np.frombuffer(fpgaOutputs[sId].get_obj(), dtype = np.float32)
    fpgaRT.exec_async( input_ptrs[sId], npout_view, sId)
    
    qWait.put((sId, img_list, shMemIdxArr))
  
  qWait.put ((None, None, None))
  #elapsedTime = ( time.time() - startTime )
  #print ( "FPGA_process: ", float(numProcessed)/elapsedTime, "img/s")
  t.join()
  xdnn.closeHandle()   
#
# This function post-processes FPGA output:
# 1) Compute the final FC + Softmax layers
# 2) Print classification & accuracy
# 
def post_process ( qFrom, args, img_paths, streamQ, fpgaOutputs):
    numProcessed = 0
    labels = xdnn_io.get_labels(args['labels'])
    zmqPub = None
    if args['zmqpub']:
      zmqPub = ZmqResultPublisher(args['deviceID'])
    goldenMap = None
    if args['golden']:
      goldenMap = xdnn_io.getGoldenMap(args['golden'])
      top5Count = 0
      top1Count = 0
      
    (fcWeight, fcBias) = xdnn_io.loadFCWeightsBias(args)
    bsz = args['batch_sz']
    fcOutput = np.empty((bsz, args['outsz'],), dtype=np.float32, order='C')
    start = 0
    while True:
      (sId, img_idx) = qFrom.get()
      if numProcessed == 0:
        start = timeit.default_timer()
      if sId is None or img_idx is None:
        break
      
      imgList = []
      for x in np.nditer(img_idx):
        if x >= 0:
          imgList.append(img_paths[x])
          numProcessed += 1
          
      npout_view = np.frombuffer(fpgaOutputs[sId].get_obj(), dtype = np.float32)
      xdnn.computeFC(fcWeight, fcBias, npout_view, bsz, args['outsz'], args['fpgaoutsz'], fcOutput)
      streamQ.put(sId)
 
      smaxOutput = xdnn.computeSoftmax(fcOutput)
      if args['golden']:
        for i,p in enumerate ( imgList ):
          top1Count += xdnn_io.isTopK(smaxOutput[i], goldenMap, p, labels, 1)
          top5Count += xdnn_io.isTopK(smaxOutput[i], goldenMap, p, labels, 5)
          
      if zmqPub is not None:
        predictMsg = xdnn_io.getClassification(smaxOutput, imgList, labels, zmqPub = True)
        zmqPub.send(predictMsg)

    print ( "%g images/s" % ( float(numProcessed) / (time.time() - start )  ))
  
    if args['golden']:
        print ("\nAverage accuracy (n=%d) Top-1: %.1f%%, Top-5: %.1f%%\n") \
          % (numProcessed, 
             float(top1Count)/float(numProcessed)*100., 
             float(top5Count)/float(numProcessed)*100.)        
    
    
class PreProcessManager(object):
  def __init__(self, args,q, img_paths, sharedInputArrs, prepProcQ):
    ret = xdnn.createManager(args['xlnxlib'])
    if ret != True:
      sys.exit(1)
    np.random.seed(123)  # for reproducibility
    self._args = args
    self._q = q
    self._imgpaths = img_paths
    current = mp.current_process()
    self._procid = (int(current._identity[0]) - 1) % args['numprepproc']
    self._sharedmem = sharedInputArrs   
    self._prepQ = prepProcQ
    
    #HWC format as this is the native format that comes out of jpeg decode
    self._meanarr = np.zeros ( (args['in_shape'][1], args['in_shape'][2], args['in_shape'][0],), dtype = np.float32, order='C' )
    self._meanarr += args['img_mean']
 
  def prepImage(self, inum):
    buf_id = self._prepQ.get()
    np_arr = np.frombuffer(self._sharedmem[buf_id].get_obj(), dtype = np.float32)
    np_arr = np.reshape ( np_arr, (1,) + self._args['in_shape'], order = 'C')
    np_arr[:], _ = xdnn_io.loadImageBlobFromFile(self._imgpaths[inum], self._args['img_raw_scale'], self._meanarr, 
                                             self._args['img_input_scale'], self._args['in_shape'][1], self._args['in_shape'][2])
    self._q.put ( (buf_id, inum) )

prep_inst = None        

def init_prepImage (args, q, img_paths, sharedInputArrs, prepProcQ):
  global prep_inst
  prep_inst = PreProcessManager(args, q, img_paths, sharedInputArrs, prepProcQ)

def run_prepImage (imgpath_idx):
  return prep_inst.prepImage(imgpath_idx)

ready_list = []
def dummy_func(index):
    global ready_list
    ready_list.append(index)
    
def main():
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--numprepproc', type=int, default=1,
                      help='number of parallel processes used to decode and quantize images')
  parser.add_argument('--numstream', type=int, default=16,
                      help='number of FPGA streams')
  parser.add_argument('--deviceID', type = int, default = 0, 
        help='FPGA no. -> FPGA ID to run in case multiple FPGAs')    
  args = parser.parse_args()
  args = xdnn_io.make_dict_args(args)
  ret = xdnn.createManager(args['xlnxlib'])
  if ret != True:
    sys.exit(1)
    
  sharedInputArrs = []
  fpgaOutputs = []
    
  qPrep = mp.Queue(maxsize=args['numprepproc']*10)
  qFpga = mp.Queue(maxsize=100)
  streamQ = mp.Queue(maxsize=args['numstream'])
  prepProcQ = mp.Queue(maxsize=100)
  for i in range( args['numstream'] ):
    shared_arr = mp.Array(ctypes.c_float, args['batch_sz'] * args['fpgaoutsz'])
    fpgaOutputs.append(shared_arr)
    streamQ.put ( i )
    
  for i in range(100):
    bufSize = np.prod(args['in_shape'])
    sharedInputArrs.append( mp.Array(ctypes.c_float, bufSize) )
    prepProcQ.put (i)
    
  img_paths = xdnn_io.getFilePaths(args['images'])

  p = mp.Pool( initializer = init_prepImage, initargs = (args, qPrep, img_paths, sharedInputArrs, prepProcQ, ), processes = args['numprepproc'])

  xdnnProc = mp.Process(target=fpga_process_async, args=(qPrep, qFpga, args, len(img_paths), sharedInputArrs,prepProcQ, streamQ, fpgaOutputs,))  
  xdnnProc.start()
  
  postProc = mp.Process(target=post_process, args=(qFpga, args, img_paths,streamQ, fpgaOutputs,))
  postProc.start()   
  if args['perpetual']: 
    while True:
      res = [p.map_async(run_prepImage, range(len(img_paths)))]
      for j in res:
        j.wait()
        del j
  else:
    p.map_async(run_prepImage, range(len(img_paths)))
  
  xdnnProc.join()
  postProc.join()
  
  p.close()
  p.join()  

if __name__ == '__main__':
    main()

