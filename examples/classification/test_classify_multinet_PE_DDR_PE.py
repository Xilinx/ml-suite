##################################################################################
# Copyright (c) 2017, Xilinx, Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
##################################################################################

#!/usr/bin/python

# Example for asynchronous multi-net classification using xdnn. Derived from test_hclassify.py
# 2017-11-09 22:53:07 parik

import argparse
import os.path
import math
import sys
import timeit
import json
import xdnn, xdnn_io
import numpy as np
import nms
import cv2
from ast import literal_eval as l_eval
from collections import defaultdict
from skimage.transform import resize

import pdb

def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.
    Parameters
    ----------
    im : (H x W x K) ndarray
    new_dims : (height, width) tuple of new dimensions.
    interp_order : interpolation order, default is linear.
    Returns
    -------
    im : resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im.astype(np.float32) - im_min) / (im_max - im_min)
            resized_std = resize(im_std, new_dims, order=interp_order, mode='constant')
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return (e_x)/(e_x.sum(axis=0,keepdims=True))

# example for multiple executors
def main(argv=None):
    args = xdnn_io.processCommandLine(argv)

    startTime = timeit.default_timer()
    ret = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", args['xlnxlib'])
    if ret != 0:
      sys.exit(1)
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to createHandle (%f ms):" % (elapsedTime * 1000)

    # we do not need other args keys except 'jsoncfg'
    args = args['jsoncfg']

    netCfgs   = defaultdict(dict)
    confNames = []
    startTime = timeit.default_timer()
    for streamId, netCfg_args in enumerate(args):
      confName        = str(netCfg_args['name'])
      confNames      += [confName]

      netCfg_args['netcfg']            = './data/{}_{}.cmd'.format(netCfg_args['net'], netCfg_args['dsp'])
      netCfgs[confName]['streamId']    = streamId
      netCfgs[confName]['args']        = netCfg_args
      (netCfgs[confName]['weightsBlobs'],
       netCfgs[confName]['fcWeights'],
       netCfgs[confName]['fcBiases'])  = xdnn_io.loadWeights( netCfg_args )
      netCfgs[confName]['batch_sz']    = 1
      netCfgs[confName]['fpgaOutputs'] = xdnn_io.prepareOutput(netCfg_args["fpgaoutsz"], netCfgs[confName]['batch_sz'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to init (%f ms):" % (elapsedTime * 1000)

    ## run YOLO
    confName = 'yolo'
    netCfg   = netCfgs[confName]

    startTime = timeit.default_timer()
    (netCfg['fpgaInputs'],
     netCfg['batch_sz'],
     netCfg['shapes'])    = xdnn_io.prepareInput(netCfg['args'], netCfg['args']['PE'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to transfer input image to FPGA (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    xdnn.exec_async(netCfg['args']['netcfg'],
                    netCfg['weightsBlobs'],
                    netCfg['fpgaInputs'],
                    netCfg['fpgaOutputs'],
                    netCfg['batch_sz'],
                    netCfg['args']['quantizecfg'],
                    netCfg['args']['scaleB'],
                    netCfg['args']['PE'],
                    netCfg['streamId'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to execute Yolo on FPGA (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    xdnn.get_result(netCfg['args']['PE'], netCfg['streamId'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to retrieve yolo outputs from FPGA (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    out_h         = \
    out_w         = netCfg['args']['in_shape'][1] / 32
    anchor_boxes  = 5
    objectness    = 1
    coordinates   = 4
    classes       = 80
    out_c         = objectness + coordinates + classes

    # Reshape the fpgaOutputs into a 4D volume
    yolo_outputs = netCfg['fpgaOutputs'].reshape(anchor_boxes, out_c, out_h, out_w)

    # Apply sigmoid to 1st, 2nd, 4th channel for all anchor boxes
    yolo_outputs[:,0:2,:,:] = sigmoid(yolo_outputs[:,0:2,:,:]) # (X,Y) Predictions
    yolo_outputs[:,4,:,:]   = sigmoid(yolo_outputs[:,4,:,:])   # Objectness / Box Confidence

    # Apply softmax on the class scores foreach anchor box
    for box in range(anchor_boxes):
        yolo_outputs[box,5:,:,:]  = softmax(yolo_outputs[box,5:,:,:])


    # Perform Non-Max Suppression
    # Non-Max Suppression filters out detections with a score lesser than 0.24
    # Additionally if there are two predections with an overlap > 30%, the prediction with the lower score will be filtered
    scorethresh = 0.24
    iouthresh   = 0.3
    bboxes = nms.do_baseline_nms(yolo_outputs.flat,
                                 netCfg['shapes'][0][1],
                                 netCfg['shapes'][0][0],
                                 netCfg['args']['in_shape'][2],
                                 netCfg['args']['in_shape'][1],
                                 out_w,
                                 out_h,
                                 anchor_boxes,
                                 classes,
                                 scorethresh,
                                 iouthresh)

    with open(netCfg['args']['labels']) as f:      
        namez = f.readlines()      
        names = [x.strip() for x in namez]
        
    # Lets print the detections our model made
    for j in range(len(bboxes)):
        print("Obj %d: %s" % (j, names[bboxes[j]['classid']]))
        print("\t score = %f" % (bboxes[j]['prob']))
        print("\t (xlo,ylo) = (%d,%d)" % (bboxes[j]['ll']['x'], bboxes[j]['ll']['y']))
        print("\t (xhi,yhi) = (%d,%d)" % (bboxes[j]['ur']['x'], bboxes[j]['ur']['y']))


    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to execute on CPU (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()

    img = cv2.imread(netCfg['args']['images'][0])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # YOLO was trained with RGB, not BGR like Caffe

    # choose one of the bounding boxes
    obj_idx = 0

    # specify a margin added to the selected bounding box
    margin  = 10

    H_slice = slice(max(0, bboxes[obj_idx]['ur']['y']-margin), min(img.shape[0], bboxes[obj_idx]['ll']['y']+margin))
    W_slice = slice(max(0, bboxes[obj_idx]['ll']['x']-margin), min(img.shape[1], bboxes[obj_idx]['ur']['x']+margin))
    img = img[H_slice, W_slice, :]

    print('pass obj {}: {} with size {} to googlenet'.format(obj_idx, names[bboxes[obj_idx]['classid']], img.shape))

    cv2.imwrite('cropped_yolo_output.jpg', img)

    '''
    if img.shape[-1] == 1 or img.shape[-1] == 3:
        # [H, W, C]
        old_dims = np.array(img.shape[:2], dtype=float)
    else:
        # [C, H, W]
        old_dims = np.array(img.shape[1:], dtype=float)
    '''

    ## run GOOGLENET
    confName = 'googlenet'
    netCfg   = netCfgs[confName]

    '''
    new_dims = netCfg['args']['in_shape']
    if new_dims[-1] == 1 or new_dims[-1] == 3:
        # [H, W, C]
        new_dims = np.array(new_dims[:2], dtype=int)
    else:
        # [C, H, W]
        new_dims = np.array(new_dims[1:], dtype=int)

    scale_dims    = new_dims.copy()
    min_scale_idx = np.argmin(old_dims/new_dims)
    if min_scale_idx == 0:
      scale_dims[1] = scale_dims[0] * old_dims[1] / old_dims[0]
    else:
      scale_dims[0] = scale_dims[1] * old_dims[0] / old_dims[1]

    scale_dims = scale_dims.astype(int)

    # transform input image to match googlenet
    # scale the image
    print('scale image to {}'.format(scale_dims))
    img = resize_image(img, list(scale_dims))
    cv2.imwrite('rescaled_scaled.jpg', img)

    # crop the image
    crop_idxs = [np.arange(new_dims[i]) + int((scale_dims[i]-new_dims[i])/2) for i in range(2)]

    if img.shape[-1] == 1 or img.shape[-1] == 3:
        # [H, W, C]
        img = img[crop_idxs[0].reshape(-1,1), crop_idxs[1], :]
    else:
        # [C, H, W]
        img = img[:, crop_idxs[0].reshape(-1,1), crop_idxs[1]]

    print('crop image to {}'.format(img.shape))
    cv2.imwrite('rescaled_cropped.jpg', img)

    #img = np.transpose(img, (2, 0, 1))
    #cv2.imwrite('rescaled_transposed.jpg', img)
    '''

    netCfg['args']['images'] = [img]
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to prepare googlenet image on CPU (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    (netCfg['fpgaInputs'],
     netCfg['batch_sz'],
     netCfg['shapes'])    = xdnn_io.prepareInput(netCfg['args'], netCfg['args']['PE'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to transfer input image to FPGA (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    xdnn.exec_async(netCfg['args']['netcfg'],
                    netCfg['weightsBlobs'],
                    netCfg['fpgaInputs'],
                    netCfg['fpgaOutputs'],
                    netCfg['batch_sz'],
                    netCfg['args']['quantizecfg'],
                    netCfg['args']['scaleB'],
                    netCfg['args']['PE'],
                    netCfg['streamId'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to execute googlenet on FPGA (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    xdnn.get_result(netCfg['args']['PE'], netCfg['streamId'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to retrieve googlenet outputs from FPGA (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    fcOut = np.empty( (netCfg['batch_sz'] * netCfg['args']['outsz']), dtype=np.float32, order = 'C')
    xdnn.computeFC(netCfg['fcWeights'],
                   netCfg['fcBiases'],
                   netCfg['fpgaOutputs'],
                   netCfg['batch_sz'],
                   netCfg['args']['outsz'],
                   netCfg['args']['fpgaoutsz'],
                   fcOut)
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to run FC layers on CPU (%f ms):" % (elapsedTime * 1000)

    startTime = timeit.default_timer()
    softmaxOut = xdnn.computeSoftmax(fcOut, netCfg['batch_sz'])
    elapsedTime = timeit.default_timer() - startTime
    print "\nTime to run Softmax on CPU (%f ms):" % (elapsedTime * 1000)

    xdnn_io.printClassification(softmaxOut, netCfg['args']);

    print "\nSuccess!\n"

    xdnn.closeHandle()

if __name__ == '__main__':
  argv = None

  '''
  import os
  import re

  XCLBIN_PATH  = os.environ['XCLBIN_PATH']
  LIBXDNN_PATH = os.environ['LIBXDNN_PATH']
  DSP_WIDTH    = 56
  BITWIDTH     = 8
  MLSUITE_ROOT = os.environ['MLSUITE_ROOT']

  argv =   "--xclbin {0}/xdnn_v2_32x{1}_{2}pe_{3}b_{4}mb_bank21.xclbin \
            --xlnxlib {5} \
            --jsoncfg data/multinet_PE_DDR_PE.json".format(XCLBIN_PATH, DSP_WIDTH, 112/DSP_WIDTH, BITWIDTH, 2+DSP_WIDTH/14, LIBXDNN_PATH)

  argv = re.split(r'(?<!,)\s+', argv)
  '''

  main(argv)

