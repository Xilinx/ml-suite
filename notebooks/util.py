#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

from __future__ import print_function

from progressbar import ProgressBar
import numpy as np
import tensorflow as tf

from xfdnn.rt.xdnn_util import make_list
from xfdnn.rt.xdnn_io import loadImageBlobFromFileScriptBase


########################################################################
## USER EDITABLE:
########################################################################
### Minimum required variables to perform preprocessing
INPUT_NODES  = 'data'
LABEL_OFFSET = 0
BATCH_SIZE   = 1

### Preprocessing formulas
### Available transformations: ['resize', 'resize2mindim', 'resize2maxdim', 'crop_letterbox',
###                             'crop_center', 'plot', 'pxlscale', 'meansub', 'chtranspose', 'chswap']

# for resnet50, inception_v1
CMD_SEQ        = [
                  ('meansub', [103.939, 116.779, 123.68]),
                  ('resize2mindim', [224, 224]),
                  ('crop_center', [224, 224]),
                 ]

# for inception_v4
# CMD_SEQ        = [
#                   ('pxlscale', 1/255.),
#                   ('meansub', 0.5),
#                   ('pxlscale', 2),
#                   ('resize2mindim', [299, 299]),
#                   ('crop_center', [299, 299]),
#                  ]

# for squeezenet
# CMD_SEQ        = [
#                   ('resize2mindim', [227, 227]),
#                   ('crop_center', [227, 227]),
#                   ('chswap',(2,1,0)),
#                   ('meansub', [104.006, 116.669, 122.679]),
#                  ]
########################################################################





# Environment Variables (obtained by running "source overlaybins/setup.sh")
IMAGEDIR   = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/"
IMAGELIST  = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt"
LABELSLIST = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"

INCLUDE_LABELS = False


def input_fn(iter):
  images = []
  labels = []
  line = open(IMAGELIST).readlines()
  for index in range(BATCH_SIZE):
    curline = line[iter * BATCH_SIZE + index].strip()
    [calib_image_name, calib_label_id] = curline.split(' ')
    labels.append(int(calib_label_id) + LABEL_OFFSET)

    image, __ = loadImageBlobFromFileScriptBase(IMAGEDIR + calib_image_name, CMD_SEQ)
    images.append(image)

  labels = np.array(labels)
  if INCLUDE_LABELS:
    return {INPUT_NODES: images, 'labels': labels}
  else:
    return {INPUT_NODES: images}

def top5_accuracy(graph, input_nodes, output_nodes, iter_cnt, batch_size, label_offset=0):
  global BATCH_SIZE, INPUT_NODES, INCLUDE_LABELS, LABEL_OFFSET

  INPUT_NODES    = input_nodes
  INCLUDE_LABELS = True
  LABEL_OFFSET   = label_offset
  BATCH_SIZE     = batch_size

  with tf.Session(graph=graph) as sess:
    input_tensors = {node: sess.graph.get_operation_by_name(node).outputs[0] for node in make_list(input_nodes)}
    output_tensor = sess.graph.get_operation_by_name(output_nodes).outputs[0]

    top1_acc = 0
    top5_acc = 0
    progress = ProgressBar()
    line = open(IMAGELIST).readlines()
    for iter in progress(range(iter_cnt)):
      inputs = input_fn(iter)
      correct_labels = inputs['labels']

      predictions = sess.run(output_tensor, feed_dict={tensor: inputs[name] for name, tensor in input_tensors.items()})

      top1_prediction = np.argmax(predictions, axis=1)
      top5_prediction = np.argsort(predictions, axis=1)[:,-5:]

      top1_accuracy = sum(top1_prediction == correct_labels)
      top5_accuracy = sum([label in top5_prediction for label in correct_labels])

      top1_acc += top1_accuracy
      top5_acc += top5_accuracy

    total_samples = float(iter_cnt*batch_size)
    final_top1_acc = top1_acc/total_samples
    final_top5_acc = top5_acc/total_samples
    print ('top1_acc:{}, top5_acc:{}'.format(final_top1_acc,final_top5_acc))




def plot_all(X):
  import numpy as np 
  import matplotlib.pyplot as plt
  
  ax = plt.subplot(221)
  ax.set_title("Vector X in FP32 Representation",fontsize=20)
  plt.setp(ax.get_xticklabels(), visible=False)
  ax.set_ylabel("Value")
  plt.plot(X["fp32"])

  bx = plt.subplot(222)
  bx.set_title("Mapping FP32 to INT8",fontsize=20)
  plt.setp(bx.get_xticklabels(), visible=False)
  bx.set_ylabel("Value")
  plt.plot(X["fp32"])
  plt.plot(X["threshold"]*np.ones_like(X["fp32"]),"k",linewidth=5.0)
  plt.plot(-1*X["threshold"]*np.ones_like(X["fp32"]),"k",linewidth=5.0)

  for i in range(128):
    plt.plot(i/X["sf"]*np.ones_like(X["fp32"]),"m",linewidth=0.1)
        
  for i in range(128):
    plt.plot(-1*i/X["sf"]*np.ones_like(X["fp32"]),"m",linewidth=0.1)

  cx = plt.subplot(223)
  cx.set_title("Vector X in INT8 Representation",fontsize=20)
  cx.set_xlabel("Element #")
  cx.set_ylabel("Value")
  plt.plot(X["int8"])

  dx = plt.subplot(224)
  dx.set_title("Percent Error by Element",fontsize=20)
  dx.set_xlabel("Element #")
  dx.set_ylabel("Percent Error")
  plt.plot(X["perror"])

  plt.subplots_adjust(top=2,bottom=0.1,left=0.1,right=4,wspace=0.2)

def plot_all2(X):
  import numpy as np 
  import matplotlib.pyplot as plt
  
  ax = plt.subplot(221)
  ax.set_title("Vector Y in FP32 Representation",fontsize=20)
  plt.setp(ax.get_xticklabels(), visible=False)
  ax.set_ylabel("Value")
  plt.plot(X["fp32"][0])

  """
  bx = plt.subplot(222)
  bx.set_title("Mapping FP32 to INT8",fontsize=20)
  plt.setp(bx.get_xticklabels(), visible=False)
  bx.set_ylabel("Value")
  plt.plot(X["fp32"][0])
  """

  cx = plt.subplot(223)
  cx.set_title("Vector Y in INT Representation",fontsize=20)
  cx.set_xlabel("Element #")
  cx.set_ylabel("Value")
  plt.plot(X["int"][0])

  dx = plt.subplot(224)
  dx.set_title("Percent Error by Element",fontsize=20)
  dx.set_xlabel("Element #")
  dx.set_ylabel("Percent Error")
  plt.plot(X["perror"][0])

  plt.subplots_adjust(top=2,bottom=0.1,left=0.1,right=4,wspace=0.2)

def findShiftScale(val):

  import numpy as np
  # val = x * 2^e
  # e must be a negative integer
  # x must be a positive integer
  e = np.ceil(np.log2(val))
  x = 1

  e_lifo = []
  x_lifo = []

  approx = x * 2**e
  delta = val-approx
  oldloss = np.square(val-approx)
  
  while True:
    approx = x * 2**e
    delta = val-approx
    loss = np.square(val-approx)

    if loss < oldloss and delta > 0:
      e_lifo.append(e)
      x_lifo.append(x)

    oldloss = loss

    if delta < 0: # Make approximation smaller
      e -= 1
      x *= 2
      x -= 1

    else:
      x += 1

    if x > 256 or e < -40:
      return e_lifo[-1],x_lifo[-1]
