#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import os,sys,timeit,json
from multiprocessing import Process, Queue

# To control print verbosity
import logging as log

# Bring in some utility functions from local file
from yolo_utils import cornersToxywh,sigmoid,softmax,generate_colors,draw_boxes
import numpy as np

# Bring in a C implementation of non-max suppression
sys.path.append('nms')
import nms

# Bring in Xilinx Caffe Compiler, and Quantizer
# We directly compile the entire graph to minimize data movement between host, and card
from xfdnn.tools.compile.frontends.frontend_caffe  import CaffeFrontend as xfdnnCompiler
from xfdnn.tools.quantize.frontends.frontend_caffe import CaffeFrontend as xfdnnQuantizer

# Bring in Xilinx XDNN middleware
from xfdnn.rt import xdnn
from xfdnn.rt import xdnn_io


class xyolo():
  def __init__(self,batch_sz=10,in_shape=[3,608,608],quantizecfg="yolo_deploy_608.json",xclbin=None,
               netcfg="yolo.cmds",datadir="yolov2.caffemodel_data",labels="coco.names",xlnxlib="libxfdnn.so",firstfpgalayer="conv0",classes=80,verbose=False):

    if verbose: 
      log.basicConfig(format="%(levelname)s: %(message)s",level=log.DEBUG)
      log.info("Running with verbose output")
    else:
      log.basicConfig(format="%(levelname)s: %(message)s")

    if xclbin is None:
      log.error("XYOLO initialized without reference to xclbin, please set this before calling detect!!")
      sys.exit(1)

    self.xdnn_handle = None 
    
    log.info("Reading labels...")
    with open(labels) as f:
      names = f.readlines()
    self.names = [x.strip() for x in names]
    
    # Arguments exposed to user
    self.in_shape    = in_shape
    self.quantizecfg = quantizecfg
    self.xclbin      = xclbin
    self.netcfg      = netcfg
    self.datadir     = datadir
    self.labels      = labels
    self.xlnxlib     = xlnxlib
    self.batch_sz    = batch_sz
    self.firstfpgalayer = firstfpgalayer # User may be using their own prototxt w/ unique names
    self.classes = classes               # User may be using their own prototxt with different region layer
  
    # Arguments not exposed to user
    ## COCO categories are not sequential
    self.img_raw_scale = "255.0"
    self.img_input_scale = "1.0"
    self.cats = [1,2,3,4,5,6,7,8,9,10,11,
                 13,14,15,16,17,18,19,20,21,22,23,24,25,
                 27,28,
                 31,32,33,34,35,36,37,38,39,40,41,42,43,44,
                 46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,
                 67,
                 70,
                 72,73,74,75,76,77,78,79,80,81,82,
                 84,85,86,87,88,89,90]
    self.images         = None
    self.scaleA         = 10000
    self.scaleB         = 30
    self.PE             = -1
    self.transform      = "yolo" # XDNN_IO will scale/letterbox the image for YOLO network
    self.img_mean       = "0,0,0"
    self.net_w          = self.in_shape[1]
    self.net_h          = self.in_shape[2]
    import math
    self.out_w = int(math.ceil(self.net_w / 32.0))
    self.out_h = int(math.ceil(self.net_h / 32.0))
    self.bboxplanes = 5
    #self.classes = 80
    self.scorethresh = 0.24
    self.iouthresh = 0.3
    self.groups = self.out_w*self.out_h
    self.coords = 4
    self.groupstride = 1
    self.batchstride = (self.groups)*(self.classes+self.coords+1)
    self.beginoffset = (self.coords+1)*(self.out_w*self.out_h)
    self.outsize = (self.out_w*self.out_h*(self.bboxplanes+self.classes))*self.bboxplanes
    self.colors = generate_colors(self.classes) # Generate color pallette for drawing boxes

    config = vars(self)

    self.q_fpga = Queue(maxsize=1)
    self.q_bbox = Queue(maxsize=1)

    if "single_proc_mode" in config:
      self.proc_fpga = None
      self.proc_bbox = None
    else:
      self.proc_fpga = Process(target=self.fpga_stage, args=(config, self.q_fpga, self.q_bbox))  
      self.proc_bbox = Process(target=self.bbox_stage, args=(config, self.q_bbox))  
      self.proc_fpga.start()
      self.proc_bbox.start()

    log.info("Running network input %dx%d and output %dx%d"%(self.net_w,self.net_h,self.out_w,self.out_h))

  def __enter__(self):
    log.info("Entering XYOLO WITH")
    return self

  def __exit__(self,*a):
    self.stop()

    if self.xdnn_handle:
      xdnn.closeHandle()

  @staticmethod
  def fpga_stage(config, q_fpga, q_bbox, maxNumIters=-1):
    config['xdnn_handle'], handles = xdnn.createHandle(config['xclbin'], "kernelSxdnn_0")
    if config['xdnn_handle'] != 0:
      log.error("Failed to start FPGA process ",
        " - could not open xclbin %s %s!" \
        % (config['xclbin'], config['xlnxlib']))
      sys.exit(1)

    fpgaRT = xdnn.XDNNFPGAOp(handles, config)

    # Allocate FPGA Outputs 
    fpgaOutSize = config['out_w']*config['out_h']*config['bboxplanes']*(config['classes']+config['coords']+1)
    fpgaOutput = np.empty((config['batch_sz'], fpgaOutSize,), dtype=np.float32, order='C')
    raw_img = np.empty(((config['batch_sz'],) + config['in_shape']), dtype=np.float32, order='C')

    numIters = 0
    while True:
      numIters += 1
      if maxNumIters > 0 and numIters > maxNumIters:
        break
      
      job = q_fpga.get()
      if job == None:
        q_bbox.put(None) # propagate 'stop' signal downstream
        sys.exit(0)

      images = job['images']
      display = job['display']
      coco = job['coco']

      if images is not None:
        log.info("Running Image(s):")
        log.info(images)
        config['images'] = images
      else:
        log.error("Detect requires images as a parameter")
        continue

      log.info("Preparing Input...")
      shapes = []
      for i,img in enumerate(images):
        raw_img[i,...], s = xdnn_io.loadYoloImageBlobFromFile(img,  config['in_shape'][1], config['in_shape'][2])
        shapes.append(s)

      job['shapes'] = shapes # pass shapes to next stage

      # EXECUTE XDNN
      log.info("Running %s image(s)"%(config['batch_sz']))
      startTime = timeit.default_timer()
      fpgaRT.execute(raw_img, fpgaOutput, config['PE'])
      elapsedTime = timeit.default_timer() - startTime

      # Only showing time for second run because first is loading script
      log.info("\nTotal FPGA: %f ms" % (elapsedTime*1000))
      log.info("Image Time: (%f ms/img):" % (elapsedTime*1000/config['batch_sz']))

      q_bbox.put((job, fpgaOutput))

  @staticmethod
  def bbox_stage(config, q_bbox, maxNumIters=-1):
    results = []

    numIters = 0
    while True:
      numIters += 1
      if maxNumIters > 0 and numIters > maxNumIters:
        break

      payload = q_bbox.get()
      if payload == None:
        break
      (job, fpgaOutput) = payload
      fpgaOutput = fpgaOutput.flatten()

      images = job['images']
      display = job['display']
      coco = job['coco']

      for i in range(config['batch_sz']):
        log.info("Results for image %d: %s"%(i, images[i]))
        startidx = i*config['outsize']
        softmaxout = fpgaOutput[startidx:startidx+config['outsize']]

        # first activate first two channels of each bbox subgroup (n)
        for b in range(config['bboxplanes']):
          for r in range(config['batchstride']*b, config['batchstride']*b+2*config['groups']):
            softmaxout[r] = sigmoid(softmaxout[r])
          for r in range(config['batchstride']*b+config['groups']*config['coords'], config['batchstride']*b+config['groups']*config['coords']+config['groups']):
            softmaxout[r] = sigmoid(softmaxout[r])
      
        # Now softmax on all classification arrays in image
        for b in range(config['bboxplanes']):
          for g in range(config['groups']):
            softmax(config['beginoffset'] + b*config['batchstride'] + g*config['groupstride'], softmaxout, softmaxout, config['classes'], config['groups'])

        # NMS
        bboxes = nms.do_baseline_nms(softmaxout, job['shapes'][i][1], job['shapes'][i][0], config['net_w'], config['net_h'], config['out_w'], config['out_h'], config['bboxplanes'], config['classes'], config['scorethresh'], config['iouthresh'])

        # REPORT BOXES
        log.info("Found %d boxes"%(len(bboxes)))
        for j in range(len(bboxes)):
          log.info("Obj %d: %s" % (j, config['names'][bboxes[j]['classid']]))
          log.info("\t score = %f" % (bboxes[j]['prob']))
          log.info("\t (xlo,ylo) = (%d,%d)" % (bboxes[j]['ll']['x'], bboxes[j]['ll']['y']))
          log.info("\t (xhi,yhi) = (%d,%d)" % (bboxes[j]['ur']['x'], bboxes[j]['ur']['y']))
          filename = images[i] 
          if coco:
            image_id = int(((filename.split("/")[-1]).split("_")[-1]).split(".")[0])
          else:
            image_id = filename.split("/")[-1]
          x,y,w,h = cornersToxywh(bboxes[j]["ll"]["x"],bboxes[j]["ll"]["y"],bboxes[j]['ur']['x'],bboxes[j]['ur']['y'])
          result = {"image_id":image_id,"category_id": config['cats'][bboxes[j]["classid"]],"bbox":[x,y,w,h],"score":round(bboxes[j]['prob'],3)}
          results.append(result)

        # DRAW BOXES w/ LABELS
        if display:
          draw_boxes(images[i],bboxes,config['names'],config['colors'])

    log.info("Saving results as results.json")
    with open("results.json","w") as fp:
      fp.write(json.dumps(results, sort_keys=True, indent=4))

  def detect(self,images=None,display=False,coco=False):
    self.q_fpga.put({
      'images': images,
      'display': display,
      'coco': coco
    })

    config = vars(self)

    if not self.proc_fpga:
      # single proc mode, no background procs, execute explicitly
      self.fpga_stage(config, self.q_fpga, self.q_bbox, 1)
      self.bbox_stage(config, self.q_bbox, 1)

  def stop(self):
    if not self.proc_fpga:
      return

    self.q_fpga.put(None)
    self.proc_fpga.join()
    self.proc_bbox.join()
    self.proc_fpga = None
    self.proc_bbox = None
  
if __name__ == '__main__':

  config = xdnn_io.processCommandLine()
   
  # Define the xyolo instance
  with xyolo(batch_sz=len(config["images"]),in_shape=eval(config["in_shape"]),quantizecfg=config["quantizecfg"],xclbin=config["xclbin"],verbose=True) as detector:
    detector.detect(config["images"])

    detector.stop()
