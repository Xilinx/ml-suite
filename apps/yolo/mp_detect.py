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
import logging as log
from yolo_utils import darknet_style_xywh, cornersToxywh,sigmoid,softmax,generate_colors,draw_boxes
sys.path.append('nms')
import nms

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
    i,img_list, sMemListIdx, shapeArr  = q.get()
    if i is None:
      break

    numProcessed += len(img_list)
    #print "numProcessed: ", numProcessed
    fpgaRT.get_result(i)
    for j in sMemListIdx:
      prepProcQ.put(j)
    qPost.put ( (i, img_list, shapeArr) )

  qPost.put ( ( None, None, None ))

def fpga_process_async (qFrom, qTo, args, num_img, sharedInputArrs, prepProcQ,  streamQ, fpgaOutputs):

  ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", [args["deviceID"]])
  if ret != 0:
    sys.exit(1)
  fpgaRT = xdnn.XDNNFPGAOp(handles, args)

  qWait = mp.Queue(maxsize=100)

  numStreams = args['numstream']
  bsz = args['batch_sz']
  input_ptrs = [[] for i in range(numStreams)]

  numProcessed = 0
  t = threading.Thread(target=xdnn_wait, args=(fpgaRT, qWait, qTo, prepProcQ, ))
  t.start()
  #startTime = time.time()
  while numProcessed < num_img or args['perpetual']:
    img_list = np.full( (bsz,), -1, dtype = np.int32 )
    sId = streamQ.get()
    input_ptrs[sId] = []
    shMemIdxArr = []
    shapeArr = []
    for j in range(bsz):
      (sMemIdx, img_idx, shape) = qFrom.get()
      numProcessed += 1
      img_list[j] = img_idx
      nparr_view = np.frombuffer(sharedInputArrs[sMemIdx].get_obj(), dtype = np.float32)
      nparr_view = nparr_view[np.newaxis, ...]
      input_ptrs[sId].append( nparr_view.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) )
      shMemIdxArr.append(sMemIdx)
      shapeArr.append(shape)
      if numProcessed == num_img:
        break

    npout_view = np.frombuffer(fpgaOutputs[sId].get_obj(), dtype = np.float32)
    fpgaRT.exec_async( input_ptrs[sId], npout_view, sId)

    qWait.put((sId, img_list, shMemIdxArr, shapeArr))

  qWait.put ((None, None, None, None))
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

    
    start = 0
    while True:
      (sId, img_idx, shapeArr) = qFrom.get()
      if numProcessed == 0:
        start = timeit.default_timer()
      if sId is None or img_idx is None:
        break

      imgList = []
      
      for x in np.nditer(img_idx):
        if x >= 0:
          imgList.append(img_paths[x])
          numProcessed += 1
          #print "psot proc post_process : ", numProcessed
          
      if args["benchmarkmode"]:
          streamQ.put(sId)
          continue
          
      fpgaOutput = np.frombuffer(fpgaOutputs[sId].get_obj(), dtype = np.float32)
      
      for i in range(args['batch_sz']):
        log.info("Results for image %d: %s"%(i, imgList[i]))
        startidx = i*args['fpgaoutsz']
        softmaxout = fpgaOutput[startidx:startidx+args['fpgaoutsz']]

        # first activate first two channels of each bbox subgroup (n)
        for b in range(args['bboxplanes']):
          for r in range(args['batchstride']*b, args['batchstride']*b+2*args['groups']):
            softmaxout[r] = sigmoid(softmaxout[r])
          for r in range(args['batchstride']*b+args['groups']*args['coords'], args['batchstride']*b+args['groups']*args['coords']+args['groups']):
            softmaxout[r] = sigmoid(softmaxout[r])
      
        # Now softmax on all classification arrays in image
        for b in range(args['bboxplanes']):
          for g in range(args['groups']):
            softmax(args['beginoffset'] + b*args['batchstride'] + g*args['groupstride'], softmaxout, softmaxout, args['outsz'], args['groups'])

        # NMS
        bboxes = nms.do_baseline_nms(softmaxout, shapeArr[i][1], shapeArr[i][0], args['in_shape'][1], args['in_shape'][2], args['out_w'], args['out_h'], args['bboxplanes'], args['outsz'], args['scorethresh'], args['iouthresh'])

        #print "image: ", imgList[i], " has num boxes detected  : ", len(bboxes)

#        for j in range(len(bboxes)):
#            print("Obj %d: %s" % (j, args['names'][bboxes[j]['classid']]))
#            print("\t score = %f" % (bboxes[j]['prob']))
#            print("\t (xlo,ylo) = (%d,%d)" % (bboxes[j]['ll']['x'], bboxes[j]['ll']['y']))
#            print("\t (xhi,yhi) = (%d,%d)" % (bboxes[j]['ur']['x'], bboxes[j]['ur']['y']))
            
      streamQ.put(sId)
    print ( "%g images/s" % ( float(numProcessed) / (timeit.default_timer() - start )  ))


class PreProcessManager(object):
  def __init__(self, args,q, img_paths, sharedInputArrs, prepProcQ):
    ret = xdnn.createManager()
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
    img_shape = [0,0,0]
    if not self._args['benchmarkmode']:
        np_arr = np.frombuffer(self._sharedmem[buf_id].get_obj(), dtype = np.float32)
        np_arr = np.reshape ( np_arr, (1,) + tuple(self._args['in_shape']), order = 'C')
        np_arr[:], img_shape = xdnn_io.loadYoloImageBlobFromFile(self._imgpaths[inum], self._args['in_shape'][1], self._args['in_shape'][2])
    
    self._q.put ( (buf_id, inum, img_shape) )

prep_inst = None

def init_prepImage (args, q, img_paths, sharedInputArrs, prepProcQ):
  global prep_inst
  prep_inst = PreProcessManager(args, q, img_paths, sharedInputArrs, prepProcQ)

def run_prepImage (imgpath_idx):
  return prep_inst.prepImage(imgpath_idx)

def main():
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--numprepproc', type=int, default=1,
                      help='number of parallel processes used to decode and quantize images')
  parser.add_argument('--numstream', type=int, default=16,
                      help='number of FPGA streams')
  parser.add_argument('--deviceID', type = int, default = 0,
        help='FPGA no. -> FPGA ID to run in case multiple FPGAs')
  parser.add_argument('--bboxplanes', type=int, default=5,
                      help='number of bboxplanes')
  parser.add_argument('--network_downscale_width', type=float, default=(32.0),
                      help='network_downscale_width')
  parser.add_argument('--network_downscale_height', type=float, default=(32.0),
                      help='network_downscale_width')
  parser.add_argument('--num_box_cordinates', type=int, default=4,
                      help='num_box_cordinates could be box x,y,width, height')
  
  parser.add_argument('--scorethresh', type=float, default=0.24,
                      help='thresohold on probability threshold')
  
  parser.add_argument('--iouthresh', type=float, default=0.3,
                      help='thresohold on iouthresh across 2 candidate detections')
  
  parser.add_argument('--benchmarkmode', type=int, default=0,
                      help='bypass pre/post processing for benchmarking')
  
  
  
  args = parser.parse_args()
  args = xdnn_io.make_dict_args(args)
  print args

  log.info("Reading labels...")
  with open(args['labels']) as f:
      names = f.readlines()
  args['names'] = [x.strip() for x in names]
    
    
  #args['netcfg'] = "work/yolo.cmds.json"
  #args['quantizecfg'] = "work/yolo_deploy_608_8b.json"
  #args['datadir'] = "work/yolov2.caffemodel_data"
  
  #args['netcfg'] = "work/yolo.cmds.json"
  #args['quantizecfg'] = "work/yolo_deploy_608_16b.json"
  #args['datadir'] = "work/yolo_v2_tiny.caffemodel_data"

  
  
  ret = xdnn.createManager()
  
  if ret != True:
    sys.exit(1)

  sharedInputArrs = []
  fpgaOutputs = []

  qPrep = mp.Queue(maxsize=args['numprepproc']*10)
  qFpga = mp.Queue(maxsize=100)
  streamQ = mp.Queue(maxsize=args['numstream'])
  prepProcQ = mp.Queue(maxsize=100)
  
  # Yolo specific
  out_w = args['in_shape'][1]/args['network_downscale_width']
  out_h = args['in_shape'][2]/args['network_downscale_height']
  args['out_w'] = int(out_w)
  args['out_h'] = int(out_h)
  args['coords'] = 4
  args['beginoffset'] = (args['coords']+1) * int(out_w * out_h)
  args['groups'] = int(out_w * out_h)
  args['batchstride'] = args['groups']*(args['outsz']+args['coords']+1)
  args['groupstride'] = 1
  
  fpgaOutSize = out_w*out_h*args['bboxplanes']*(args['outsz']+args['coords']+1)
  args['fpgaoutsz'] = int(fpgaOutSize)
  
  print(fpgaOutSize, args['batch_sz'])
  for i in range( args['numstream'] ):
      shared_arr = mp.Array(ctypes.c_float, int(args['batch_sz'] * fpgaOutSize))
      fpgaOutputs.append(shared_arr)
      streamQ.put ( i )

  for i in range(100):
      bufSize = np.prod(tuple(args['in_shape']))
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

