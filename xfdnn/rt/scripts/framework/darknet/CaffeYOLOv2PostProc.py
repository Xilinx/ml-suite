#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env python

import caffe
import numpy as np

class CaffeYOLOv2PostProc(caffe.Layer):

  # Called once when the network is initialized
  def setup(self, bottom, top):
    self._args = eval(self.param_str) # Get args from prototxt
    # This layer will not change the shape of the blob
    # Declare an output blob which is the same shape as the input
    top[0].reshape(*bottom[0].data.shape)

  # Called before every forward
  def reshape(self, bottom, top):
    pass

  def sigmoid(self,x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

  def softmax(self,x):
    e_x = np.exp(x-np.max(x))
    return (e_x)/(e_x.sum(axis=1,keepdims=True))
  
  # Called for every batch
  def forward(self, bottom, top):
    shape = bottom[0].data.shape
    batch_size = shape[0]
    anchor_boxes = self._args["anchor_boxes"]
    chan_per_box = 1 + 4 + self._args["classes"]
    height = shape[2]
    width = shape[3]
    blob = bottom[0].data
    blob = blob.reshape(batch_size,anchor_boxes,chan_per_box,height,width)

    # Apply sigmoid to 1st, 2nd, 4th channel for all batches, and anchor boxes
    blob[:,:,0:2,:,:] = self.sigmoid(blob[:,:,0:2,:,:]) # (X,Y) Predictions are squashed between 0,1
    blob[:,:,4,:,:]   = self.sigmoid(blob[:,:,4,:,:])   # Objectness / Box Confidence, is squashed to a probability between 0,1

    # Apply softmax on the class scores foreach anchor box in each batch
    for batch in range(batch_size):  
      for box in range(anchor_boxes):
        blob[batch,box,5:,:,:]  = self.softmax(blob[batch,box,5:,:,:])

    top[0].data[...] = blob.reshape(shape)

  def backward(self, top, propagate_down, bottom):
    raise Exception("Can't do backward propagation... yet")
