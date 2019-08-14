#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import os, sys
import timeit
import numpy as np
import multiprocessing as mp
import ctypes
import threading
import time
import logging as log
from yolo_utils import darknet_style_xywh, cornersToxywh,sigmoid,softmax,generate_colors,draw_boxes
from get_mAP_darknet import calc_detector_mAP
from detect_api_yolov3 import det_postprocess
sys.path.append('nms')
import nms

from xfdnn.rt import xdnn, xdnn_io

sys.path.insert(0, os.environ["MLSUITE_ROOT"] + '/examples/deployment_modes')
import mp_classify as mp_classify
sys.path.insert(0, os.environ["MLSUITE_ROOT"] + '/apps/yolo')

class YoloPreProcess(mp_classify.UserPreProcess):
  def run(self, inum):
    write_slot = self._shared_trans_arrs.openWriteId()
    write_arrs = self._shared_trans_arrs.accessNumpyBuffer(write_slot)
    
    if not self._args['benchmarkmode']:
      write_arrs[0][:], ishape = xdnn_io.loadYoloImageBlobFromFile(self._imgpaths[inum], self._firstInputShape[2], self._firstInputShape[3])
      write_arrs[-1][1:4] = ishape
      
    write_arrs[-1][0] = inum
    self._shared_trans_arrs.closeWriteId(write_slot)    
  
class YoloPostProcess(mp_classify.UserPostProcess):
  def loop(self):
    fpgaOutputShapes = []
    for idx in range(len( self.output_shapes)):
        fpgaOutputShape_l = self.output_shapes[idx] 
        fpgaOutputShape_l[0] = self.args['batch_sz']
        fpgaOutputShapes.append(fpgaOutputShape_l)

    while True:
      read_slot = self._shared_output_arrs.openReadId()
      if read_slot is None:
          break
      
      read_slot_arrs = self._shared_output_arrs.accessNumpyBuffer(read_slot)  
      imgList = []
      shape_list = []
      #image_id = self._qFrom.get()
      num_images = (read_slot_arrs[-1].shape)[0]
      for image_num in range(num_images):
          image_id = read_slot_arrs[-1][image_num][0]
          
          if image_id == -1:
              break
          imgList.append(self.img_paths[int(image_id)])
          shape_list.append(read_slot_arrs[-1][image_num][1:4])
      
      if self.args["benchmarkmode"]:
        self.numProcessed += len(imgList)
        #self.streamQ.put(sId)
        self._shared_output_arrs.closeReadId(read_slot)        
        continue
    
      self.run(imgList,read_slot_arrs[0:-1], fpgaOutputShapes, shape_list)  
      self._shared_output_arrs.closeReadId(read_slot)   
      
    self.finish()

  def run(self, imgList, fpgaOutput_list, fpgaOutputShape_list, shapeArr):

    if self.numProcessed == 0:
      self.startTime = timeit.default_timer()
      self.labels = xdnn_io.get_labels(self.args['labels'])
      self.zmqPub = None
      if self.args['zmqpub']:
        self.zmqPub = mp_classify.ZmqResultPublisher(self.args['deviceID'])
      self.goldenMap = None


    self.numProcessed += len(imgList)

    firstInputShape = xdnn.CompilerJsonParser(self.args['netcfg']).getInputs().itervalues().next()
    
    if((args['yolo_model'] == 'standard_yolo_v3') or (args['yolo_model'] == 'tiny_yolo_v3')):
        num_ouptut_layers= len(fpgaOutput_list)
        fpgaOutput = []
        for idx in range(num_ouptut_layers):
            fpgaOutput.append(np.frombuffer(fpgaOutput_list[idx], dtype=np.float32).reshape(tuple(fpgaOutputShape_list[idx])))
        bboxlist_for_images = det_postprocess(fpgaOutput, args, shapeArr)
        
        for i in range(min(self.args['batch_sz'], len(shapeArr))):
            print "image: ", imgList[i], " has num boxes detected  : ", len(bboxlist_for_images[i])
    
    else:        
            
        fpgaOutput  = fpgaOutput_list[0]
        fpgaOutputShape  = fpgaOutputShape_list[0]
        npout_view = np.frombuffer(fpgaOutput, dtype=np.float32)\
          .reshape(tuple(fpgaOutputShape))
        npout_view = npout_view.flatten() 
        fpgaoutsz = fpgaOutputShape[1]*fpgaOutputShape[2]*fpgaOutputShape[3]
        bboxlist_for_images = []
        for i in range(min(self.args['batch_sz'], len(shapeArr))):
          startidx = i*fpgaoutsz
          softmaxout = npout_view[startidx:startidx+fpgaoutsz]
        
          # first activate first two channels of each bbox subgroup (n)
          for b in range(self.args['bboxplanes']):
            for r in range(\
              self.args['batchstride']*b, 
              self.args['batchstride']*b+2*self.args['groups']):
              softmaxout[r] = sigmoid(softmaxout[r])
    
            for r in range(\
              self.args['batchstride']*b\
                +self.args['groups']*self.args['coords'], 
              self.args['batchstride']*b\
                +self.args['groups']*self.args['coords']+self.args['groups']):
              softmaxout[r] = sigmoid(softmaxout[r])
        
          # Now softmax on all classification arrays in image
          for b in range(self.args['bboxplanes']):
            for g in range(self.args['groups']):
              softmax(self.args['beginoffset'] + b*self.args['batchstride'] + g*self.args['groupstride'], softmaxout, softmaxout, self.args['outsz'], self.args['groups'])
    
          # NMS
          bboxes = nms.do_baseline_nms(softmaxout, shapeArr[i][1], shapeArr[i][0], firstInputShape[2], firstInputShape[3], self.args['out_w'], self.args['out_h'], self.args['bboxplanes'], self.args['outsz'], self.args['scorethresh'], self.args['iouthresh'])
          bboxlist_for_images.append(bboxes)
          print "image: ", imgList[i], " has num boxes detected  : ", len(bboxes)
          
    if self.args['golden'] is None:
        return
    
    for i in range(min(self.args['batch_sz'], len(shapeArr))):
        filename = imgList[i]
        out_file_txt = ((filename.split("/")[-1]).split(".")[0])
        out_file_txt = self.args['detection_labels']+"/"+out_file_txt+".txt"
        out_line_list = []
        bboxes = bboxlist_for_images[i]
        for j in range(len(bboxes)):
            x,y,w,h = darknet_style_xywh(shapeArr[i][1], shapeArr[i][0], bboxes[j]["ll"]["x"],bboxes[j]["ll"]["y"],bboxes[j]['ur']['x'],bboxes[j]['ur']['y'])

                  
            line_string = str(bboxes[j]["classid"])
            line_string = line_string+" "+str(round(bboxes[j]['prob'],3))
            line_string = line_string+" "+str(x)
            line_string = line_string+" "+str(y)
            line_string = line_string+" "+str(w)
            line_string = line_string+" "+str(h)
            out_line_list.append(line_string+"\n")


        log.info("writing this into prediction file at %s"%(out_file_txt))
        with open(out_file_txt, "w") as the_file:
            for lines in out_line_list:
                the_file.write(lines)

    

  def finish(self):
    print ( "[XDNN] Total time in sec: %g " % ( (timeit.default_timer() - self.startTime) ))
    print   "[XDNN] Total Images Processed : ", self.numProcessed
    print ( "[XDNN] Throughput: %g images/s" % ( float(self.numProcessed) / (timeit.default_timer() - self.startTime )  ))
    
    with open(self.args['labels']) as f:
        names = f.readlines()
    
    class_names = [x.strip() for x in names]
    
    # If ground truth labels are provided as signalled by arg golden calculate mAP score
    if self.args['golden'] is not None:
        print("Computing mAP score  :")
        print("Class names are  :", class_names)
        
        mAP = calc_detector_mAP(self.args['detection_labels'], self.args['golden'], len(class_names), class_names, self.args['prob_threshold'], self.args['iouthresh'])

mp_classify.register_pre(YoloPreProcess)
mp_classify.register_post(YoloPostProcess)

if __name__ == '__main__':
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--numprepproc', type=int, default=1,
                      help='number of parallel processes used to decode and quantize images')
  parser.add_argument('--numstream', type=int, default=6,
                      help='number of FPGA streams')
  parser.add_argument('--deviceID', type = int, default = 0,
        help='FPGA no. -> FPGA ID to run in case multiple FPGAs')
  parser.add_argument('--network_downscale_width', type=float, default=(32.0),
                      help='network_downscale_width')
  parser.add_argument('--network_downscale_height', type=float, default=(32.0),
                      help='network_downscale_width')
  parser.add_argument('--num_box_cordinates', type=int, default=4,
                      help='num_box_cordinates could be box x,y,width, height')
  
  parser.add_argument('--scorethresh', type=float, default=0.005,
                      help='thresohold on probability threshold')
  
  parser.add_argument('--iouthresh', type=float, default=0.3,
                      help='thresohold on iouthresh across 2 candidate detections')
  
  parser.add_argument('--benchmarkmode', type=int, default=0,
                      help='bypass pre/post processing for benchmarking')
  parser.add_argument("--yolo_model",  type=str, default='xilinx_yolo_v2')
  parser.add_argument('--in_shape', default=[3,224,224], nargs=3, type=int, help='input dimensions') 
  
  parser.add_argument('--anchorCnt', type=int, default=5,
                      help='thresohold on iouthresh across 2 candidate detections')
  parser.add_argument('--detection_labels', help="direcotry path detected lable files in darknet style",
                        default=None, type=str, metavar="FILE")
  parser.add_argument('--prob_threshold', type=float, default=0.1, help='threshold for calculation of f1 score')

  args = parser.parse_args()
  args = xdnn_io.make_dict_args(args)
  compilerJSONObj = xdnn.CompilerJsonParser(args['netcfg'])
  firstInputShape = compilerJSONObj.getInputs().itervalues().next()
  firstOutputShape = compilerJSONObj.getOutputs().itervalues().next()
  out_w = firstOutputShape[2]
  out_h = firstOutputShape[3]

  args['net_w'] = int(firstInputShape[2])
  args['net_h'] = int(firstInputShape[3])   
  args['out_w'] = int(out_w)
  args['out_h'] = int(out_h)
  args['coords'] = 4
  args['beginoffset'] = (args['coords']+1) * int(out_w * out_h)
  args['groups'] = int(out_w * out_h)
  args['batchstride'] = args['groups']*(args['outsz']+args['coords']+1)
  args['groupstride'] = 1
  args['classes'] = args['outsz']
  args['bboxplanes'] = args['anchorCnt']
  
  print "running yolo_model : ", args['yolo_model']
      
      

  mp_classify.run(args)
