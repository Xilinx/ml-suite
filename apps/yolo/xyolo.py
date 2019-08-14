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
from yolo_utils import darknet_style_xywh, cornersToxywh,sigmoid,softmax,generate_colors,draw_boxes, process_all_yolo_layers, apply_nms
import numpy as np

# Bring in a C implementation of non-max suppression
sys.path.append('nms')
import nms

# Bring in Xilinx Caffe Compiler, and Quantizer
# We directly compile the entire graph to minimize data movement between host, and card
from xfdnn.tools.compile.bin.xfdnn_compiler_caffe import CaffeFrontend as xfdnnCompiler
#from xfdnn.tools.quantize.quantize import CaffeFrontend as xfdnnQuantizer

# Bring in Xilinx XDNN middleware
from xfdnn.rt import xdnn
from xfdnn.rt import xdnn_io

import caffe


                           
   
def correct_region_boxes(boxes_array, x_idx, y_idx, w_idx, h_idx, w, h, net_w, net_h):
    
    new_w = 0;
    new_h = 0;
    #print "x_idx, y_idx, w_idx, h_idx, w, h, net_w, net_h", x_idx, y_idx, w_idx, h_idx, w, h, net_w, net_h
    if ((float(net_w) / float(w)) < (float(net_h) / float(h))) :
        new_w = net_w
        new_h = (h * net_w) / w
    else:
        new_w = (w * net_h) / h;
        new_h = net_h
    
    boxes_array[:,x_idx] = (boxes_array[:,x_idx] - (net_w - new_w) / 2.0 / net_w) / (float(new_w) / net_w);
    boxes_array[:,y_idx] = (boxes_array[:,y_idx] - (net_h - new_h) / 2.0 / net_h) / (float(new_h) / net_h);
    boxes_array[:,w_idx] *= float(net_w) / float(new_w);
    boxes_array[:,h_idx] *= float(net_h) / float(new_h);
    
    return boxes_array
  
    

    

def darknet_maxpool_k2x2_s1(data_in, data_out):
    w = data_in.shape[2]
    h = data_in.shape[3]
    #print data_in.shape, w, h
    
    for x in range(w):
        for y in range(h):
            end_val_x = min(x+1, w-1)
            end_val_y = min(y+1, h-1)
            data_out[:,:, y,x] = np.maximum(data_in[:,:,y,end_val_x],np.maximum(data_in[:,:,end_val_y,x],np.maximum(data_in[:,:,y,x],data_in[:,:,end_val_y,end_val_x])))

class xyolo():
  def __init__(self,batch_sz=10,in_shape=[3,608,608],quantizecfg="yolo_deploy_608.json",xclbin=None,
               netcfg="yolo.cmds",weights="yolov2.caffemodel_data",labels="coco.names",xlnxlib="libxfdnn.so",firstfpgalayer="conv0",
               classes=80,
               anchor_count=5,
               score_threshold=0.2,
               iou_threshold=0.45,
               verbose=False,
               yolo_model=None,
             caffe_prototxt=None,
             caffe_model=None):

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
    
    if os.path.isdir('./out_labels') is False:
        os.makedirs('./out_labels')
        
    self.out_labels_path = './out_labels'
    
    # Arguments exposed to user
    self.in_shape    = in_shape
    self.quantizecfg = quantizecfg
    self.xclbin      = xclbin
    self.netcfg      = netcfg
    self.weights     = weights
    self.labels      = labels
    self.xlnxlib     = xlnxlib
    
    self.yolo_model         = yolo_model
    self.caffe_prototxt     = caffe_prototxt
    self.caffe_model        = caffe_model
    
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
    self.bboxplanes = anchor_count
    self.anchorCnt = anchor_count
    #self.classes = 80
    self.scorethresh = score_threshold  #0.24
    self.iouthresh   = iou_threshold    #0.45
    self.groups = self.out_w*self.out_h
    self.coords = 4
    self.groupstride = 1
    self.batchstride = (self.groups)*(self.classes+self.coords+1)
    self.beginoffset = (self.coords+1)*(self.out_w*self.out_h)
    self.outsize = (self.out_w*self.out_h*(self.coords + 1 +self.classes))*self.bboxplanes
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

    fpgaInput = fpgaRT.getInputs()
    fpgaOutput = fpgaRT.getOutputs()


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

    
      if((config['yolo_model'] == 'xilinx_yolo_v2') or (config['yolo_model'] == 'xilinx_prelu_yolo_v2') or (config['yolo_model'] == 'tiny_yolo_v2_voc')) :
          pass
      else:
          
          out_data_shape=[] 
          net = caffe.Net(config['caffe_prototxt'], config['caffe_model'], caffe.TEST)
          
          
          if(config['yolo_model'] == 'standard_yolo_v2'):
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer31-conv'].data.shape[1:4]))
          
          elif(config['yolo_model'] == 'tiny_yolo_v2'):
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer15-conv'].data.shape[1:4]))

          elif(config['yolo_model'] == 'tiny_yolo_v3'):
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer14-conv'].data.shape[1:4]))
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer21-conv'].data.shape[1:4]))

              
          elif(config['yolo_model'] == 'standard_yolo_v3'):
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer81-conv'].data.shape[1:4]))
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer93-conv'].data.shape[1:4]))
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer105-conv'].data.shape[1:4]))
          
          elif(config['yolo_model'] == 'spp_yolo_v3'):
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer88-conv'].data.shape[1:4]))
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer100-conv'].data.shape[1:4]))
              out_data_shape.append((config['batch_sz'] ,) + tuple(net.blobs['layer112-conv'].data.shape[1:4]))

          #print "out_data_shape : ", out_data_shape
          softmaxOut=[]
          for list_idx in range(len(out_data_shape)):
              softmaxOut.append(np.empty(out_data_shape[list_idx]))


      firstInput = fpgaInput.itervalues().next()
      firstOutput = fpgaOutput.itervalues().next()
      maxpool_out = np.empty_like(firstOutput)

      log.info("Preparing Input...")
      shapes = []
      inputs = []
      for i,img in enumerate(images):
        firstInput[i,...], s = xdnn_io.loadYoloImageBlobFromFile(img,  config['in_shape'][1], config['in_shape'][2])
        shapes.append(s)

      job['shapes'] = shapes # pass shapes to next stage
      # EXECUTE XDNN
      log.info("Running %s image(s)"%(config['batch_sz']))
      
      if((config['yolo_model'] == 'xilinx_yolo_v2') or (config['yolo_model'] == 'xilinx_prelu_yolo_v2') or (config['yolo_model'] == 'tiny_yolo_v2_voc')) :
          startTime = timeit.default_timer()
          fpgaRT.execute(fpgaInput, fpgaOutput, config['PE'] )
          elapsedTime = timeit.default_timer() - startTime
           
          # Only showing time for second run because first is loading script
          log.info("\nTotal FPGA: %f ms" % (elapsedTime*1000))
          log.info("Image Time: (%f ms/img):" % (elapsedTime*1000/config['batch_sz']))
           
          q_bbox.put((job, firstOutput))
           
      elif(config['yolo_model'] == 'standard_yolo_v2'):
          
          startTime = timeit.default_timer()
          fpgaRT.execute(fpgaInput, fpgaOutput, config['PE'])
          elapsedTime = timeit.default_timer() - startTime
          #out_data_shape = (config['batch_sz'] ,) + tuple(net.blobs['layer31-conv'].data.shape[1:4])
          #softmaxOut = np.empty(out_data_shape)
        
          startTime = timeit.default_timer()
          for bt_idx in range(config['batch_sz']):
              net.blobs['layer25-conv'].data[...] = fpgaOutput['layer25-conv'][bt_idx,...]
              net.blobs['layer27-conv'].data[...] = fpgaOutput['layer27-conv'][bt_idx,...]
              net.forward(start='layer28-reorg', end='layer31-conv')
              final_out = net.blobs['layer31-conv'].data[...]
              softmaxOut[0][bt_idx,...] = final_out[...]
           
          elapsedTime_cpu = timeit.default_timer() - startTime
          # Only showing time for second run because first is loading script
          print (elapsedTime*1000, (elapsedTime_cpu*1000) , ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))
          log.info("\nTotal FPGA: %f ms" % (elapsedTime*1000))
          log.info("\nTotal FPGA: %f ms" % (elapsedTime_cpu*1000))
          log.info("Image Time: (%f ms/img):" % ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))
           
          q_bbox.put((job, softmaxOut[0]))

      elif(config['yolo_model'] =='tiny_yolo_v3'):
          startTime = timeit.default_timer()
          fpgaRT.execute(fpgaInput, fpgaOutput, config['PE'])
          elapsedTime = timeit.default_timer() - startTime
          for bt_idx in range(config['batch_sz']):
                 softmaxOut[0][bt_idx,...] = fpgaOutput['layer14-conv'][bt_idx,...]
                 softmaxOut[1][bt_idx,...] = fpgaOutput['layer21-conv'][bt_idx,...]
          
          q_bbox.put((job, softmaxOut)) 
         

      elif(config['yolo_model'] == 'standard_yolo_v3'):
	  use_fpga = 1
          if (use_fpga== 1) :
	          startTime = timeit.default_timer()
        	  fpgaRT.execute(fpgaInput, fpgaOutput, config['PE'])
	          elapsedTime = timeit.default_timer() - startTime

	          startTime = timeit.default_timer()
        	  for bt_idx in range(config['batch_sz']):
                	  softmaxOut[0][bt_idx,...] = fpgaOutput['layer81-conv'][bt_idx,...]
	                  softmaxOut[1][bt_idx,...] = fpgaOutput['layer93-conv'][bt_idx,...]
        	          softmaxOut[2][bt_idx,...] = fpgaOutput['layer105-conv'][bt_idx,...]
                  
          	  elapsedTime_cpu = timeit.default_timer() - startTime
                  
                  print (elapsedTime*1000, (elapsedTime_cpu*1000) , ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))
          else:
        
		for bt_idx in range(config['batch_sz']):
	        	net.blobs['data'].data[...] = firstInput[bt_idx,...]
        	        net.forward()
                        softmaxOut[0][bt_idx,...] = net.blobs['layer81-conv'].data[...]
                        softmaxOut[1][bt_idx,...] = net.blobs['layer93-conv'].data[...]
                        softmaxOut[2][bt_idx,...] = net.blobs['layer105-conv'].data[...]
          # Only showing time for second run because first is loading script
          #log.info("\nTotal FPGA: %f ms" % (elapsedTime*1000))
          #log.info("\nTotal FPGA: %f ms" % (elapsedTime_cpu*1000))
          #log.info("Image Time: (%f ms/img):" % ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))

          q_bbox.put((job, softmaxOut))
           
      elif(config['yolo_model'] == 'spp_yolo_v3'):
          startTime = timeit.default_timer()
          fpgaRT.execute(fpgaInput, fpgaOutput, config['PE'])
          elapsedTime = timeit.default_timer() - startTime


          startTime = timeit.default_timer()
          for bt_idx in range(config['batch_sz']):
                  softmaxOut[0][bt_idx,...] = fpgaOutput['layer88-conv'][bt_idx,...]
                  softmaxOut[1][bt_idx,...] = fpgaOutput['layer100-conv'][bt_idx,...]
                  softmaxOut[2][bt_idx,...] = fpgaOutput['layer112-conv'][bt_idx,...]



          elapsedTime_cpu = timeit.default_timer() - startTime
          # Only showing time for second run because first is loading script
          print (elapsedTime*1000, (elapsedTime_cpu*1000) , ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))
          log.info("\nTotal FPGA: %f ms" % (elapsedTime*1000))
          log.info("\nTotal FPGA: %f ms" % (elapsedTime_cpu*1000))
          log.info("Image Time: (%f ms/img):" % ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))

          q_bbox.put((job, softmaxOut))

      elif(config['yolo_model'] =='tiny_yolo_v2'):
          startTime = timeit.default_timer()
          fpgaRT.execute(fpgaInput, fpgaOutput, config['PE'])
          elapsedTime = timeit.default_timer() - startTime
          darknet_maxpool_k2x2_s1(firstOutput, maxpool_out)
          
          for bt_idx in range(config['batch_sz']):
              net.blobs['data'].data[...] =  maxpool_out[bt_idx,...]
              net.forward()
              final_out = net.blobs['layer15-conv'].data[...]
              softmaxOut[0][bt_idx,...] = final_out[...]	
          elapsedTime_cpu = timeit.default_timer() - startTime
           
          print (elapsedTime*1000, (elapsedTime_cpu*1000) , ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))
          log.info("\nTotal FPGA: %f ms" % (elapsedTime*1000))
          log.info("\nTotal FPGA: %f ms" % (elapsedTime_cpu*1000))
          log.info("Image Time: (%f ms/img):" % ((elapsedTime+elapsedTime_cpu)*1000/config['batch_sz']))
           
          q_bbox.put((job, softmaxOut[0]))
           
      else:
          print("model not supported")
           

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
      
      images = job['images']
      display = job['display']
      coco = job['coco']
      
     
      if((config['yolo_model'] =='standard_yolo_v3') or (config['yolo_model'] =='tiny_yolo_v3') or (config['yolo_model'] =='spp_yolo_v3')):
          anchorCnt = config['anchorCnt']
          classes = config['classes']
          
          if (config['yolo_model'] =='tiny_yolo_v3') :
              classes = 80
              #config['classes'] = 3   
          #print "classes fpgaOutput len", classes, len(fpgaOutput)
          out_yolo_layers = process_all_yolo_layers(fpgaOutput, classes, anchorCnt, config['net_w'], config['net_h'])
          
          num_proposals_layer=[0]
          total_proposals = 0
          for layr_idx in range (len(out_yolo_layers)):
              yolo_layer_shape = out_yolo_layers[layr_idx].shape
              #print "layr_idx , yolo_layer_shape", layr_idx , yolo_layer_shape
              out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].reshape(yolo_layer_shape[0], anchorCnt, (5+classes), yolo_layer_shape[2]*yolo_layer_shape[3])
              out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].transpose(0,3,1,2)
              out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].reshape(yolo_layer_shape[0],yolo_layer_shape[2]*yolo_layer_shape[3] * anchorCnt, (5+classes))           
              #print "layr_idx, final in layer sape, outlayer shape", layr_idx, yolo_layer_shape, out_yolo_layers[layr_idx].shape
              total_proposals += yolo_layer_shape[2]*yolo_layer_shape[3] * anchorCnt
              num_proposals_layer.append(total_proposals)
              
         
          boxes_array = np.empty([config['batch_sz'], total_proposals, (5+classes)]) 
          
          for layr_idx in range (len(out_yolo_layers)):
              proposal_st = num_proposals_layer[layr_idx]
              proposal_ed = num_proposals_layer[layr_idx + 1]
              #print "proposal_st proposal_ed", proposal_st, proposal_ed
              boxes_array[:,proposal_st:proposal_ed,:] = out_yolo_layers[layr_idx][...]
              
          
          for i in range(config['batch_sz']):
              boxes_array[i,:,:] = correct_region_boxes(boxes_array[i,:,:], 0, 1, 2, 3, float(job['shapes'][i][1]), float(job['shapes'][i][0]), float(config['net_w']), float(config['net_h']))
              detected_boxes = apply_nms(boxes_array[i,:,:], classes, config['scorethresh'], config['iouthresh'])
              
              bboxes=[]
              for det_idx in range(len(detected_boxes)):
                  #print  detected_boxes[det_idx][0], detected_boxes[det_idx][1], detected_boxes[det_idx][2], detected_boxes[det_idx][3], config['names'][detected_boxes[det_idx][4]], detected_boxes[det_idx][5]
                  
                  bboxes.append({'classid' : detected_boxes[det_idx][4],
                                   'prob' : detected_boxes[det_idx][5],
                                   'll' : {'x' : int((detected_boxes[det_idx][0] - 0.5 *detected_boxes[det_idx][2]) * job['shapes'][i][1]),
                                           'y' : int((detected_boxes[det_idx][1] + 0.5 *detected_boxes[det_idx][3]) * job['shapes'][i][0])},
                                   'ur' : {'x' : int((detected_boxes[det_idx][0] + 0.5 *detected_boxes[det_idx][2]) * job['shapes'][i][1]),
                                           'y' : int((detected_boxes[det_idx][1] - 0.5 *detected_boxes[det_idx][3]) * job['shapes'][i][0])}})
    
                  log.info("Obj %d: %s" % (det_idx, config['names'][bboxes[det_idx]['classid']]))
                  log.info("\t score = %f" % (bboxes[det_idx]['prob']))
                  log.info("\t (xlo,ylo) = (%d,%d)" % (bboxes[det_idx]['ll']['x'], bboxes[det_idx]['ll']['y']))
                  log.info("\t (xhi,yhi) = (%d,%d)" % (bboxes[det_idx]['ur']['x'], bboxes[det_idx]['ur']['y']))

                  
              if display:
                  draw_boxes(images[i],bboxes,config['names'],config['colors'])
  
              filename = images[i]
              out_file_txt = ((filename.split("/")[-1]).split(".")[0])
              out_file_txt = config['out_labels_path']+"/"+out_file_txt+".txt"
              out_line_list = []
              
              for j in range(len(bboxes)):
                  #x,y,w,h = darknet_style_xywh(job['shapes'][i][1], job['shapes'][i][0], bboxes[j]["ll"]["x"],bboxes[j]["ll"]["y"],bboxes[j]['ur']['x'],bboxes[j]['ur']['y'])
                  x = detected_boxes[j][0]
                  y = detected_boxes[j][1]
                  w = detected_boxes[j][2]
                  h = detected_boxes[j][3]
                  
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

          continue
          
      
      fpgaOutput = fpgaOutput.flatten()  
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
        filename = images[i]
        out_file_txt = ((filename.split("/")[-1]).split(".")[0])
        out_file_txt = config['out_labels_path']+"/"+out_file_txt+".txt"
        
        out_line_list = []

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
          x,y,w,h = darknet_style_xywh(job['shapes'][i][1], job['shapes'][i][0], bboxes[j]["ll"]["x"],bboxes[j]["ll"]["y"],bboxes[j]['ur']['x'],bboxes[j]['ur']['y'])
          line_string = str(bboxes[j]["classid"])
          line_string = line_string+" "+str(round(bboxes[j]['prob'],3))
          line_string = line_string+" "+str(x)
          line_string = line_string+" "+str(y)
          line_string = line_string+" "+str(w)
          line_string = line_string+" "+str(h)	
          out_line_list.append(line_string+"\n")	

        # DRAW BOXES w/ LABELS
        if display:
          draw_boxes(images[i],bboxes,config['names'],config['colors'])

        log.info("writing this into prediction file at %s"%(out_file_txt))
        with open(out_file_txt, "w") as the_file:
            
            for lines in out_line_list:
                
                the_file.write(lines)


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
