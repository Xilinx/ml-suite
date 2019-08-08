#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/python

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
