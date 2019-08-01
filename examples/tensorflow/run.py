#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/python

from __future__ import print_function

import os, sys, argparse
import subprocess

from IPython.display import Image as display
from ipywidgets import interact

from progressbar import ProgressBar
import tensorflow as tf
import numpy as np
import cv2

from tensorflow.contrib.decent_q.python.input_fn import *

# Bring in ml-suite Quantizer, Compiler, and Partitioner
from xfdnn.rt.xdnn_rt_tf import TFxdnnRT as xdnnRT
from xfdnn.rt.xdnn_util import dict2attr, make_list
from xfdnn.rt.xdnn_io import default_xdnn_arg_parser


# Environment Variables (obtained by running "source overlaybins/setup.sh")
IMAGEDIR   = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/"
IMAGELIST  = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt"
LABELSLIST = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"
XCLBIN     = '/opt/ml-suite' + '/overlaybins/' + os.getenv('MLSUITE_PLATFORM','alveo-u200') + '/overlay_4.xclbin'


QUANT_BATCH_SIZE = 10

def get_default_compiler_args():
    return {
        'dsp':                  96,
        'memory':               9,
        'bytesperpixels':       1,
        'ddr':                  256,
        'data_format':          'NHWC',
        'mixmemorystrategy':    True,
        'noreplication':        True,
        'xdnnv3':               True,
        'usedeephi':            True,
        'quantz':               ''
    }


def calib_input(iter, batch_size=QUANT_BATCH_SIZE, **kwarg):
  height_out, width_out = 224, 128
  means = (123, 117, 104)
  images = []
  labels = []
  line = open(IMAGELIST).readlines()
  for index in range(0, batch_size):
    curline = line[iter * batch_size + index].strip()
    [calib_image_name, calib_label_id] = curline.split(' ')

    image = cv2.imread(IMAGEDIR + calib_image_name)
    width_in = image.shape[1]
    height_in = image.shape[0]

    scale = max(float(height_out) / height_in, float(width_out) / width_in)

    image = cv2.resize(image, (round(scale * width_in), round(s * height_in)))
    width_in = image.shape[1]
    height_in = image.shape[0]

    width_begin = round(0.5 * (width_in - width_out))
    height_begin = round(0.5 * (height_in - height_out))
    width_end = width_begin + width_out
    height_end = height_begin + height_out
    image = image[width_begin:width_end, height_begin:height_end]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image -= means
    images.append(image)
    labels.append(int(calib_label_id) + 1)
  # dict key below should be the name of graph inputs
  return {"data": images}


def preprocess(image, means, **kwarg):
  input_height, input_width = 224, 224

  ## Image preprocessing using numpy
  img  = cv2.imread(image).astype(np.float32)
  img -= np.array(make_list(means)).reshape(-1,3).astype(np.float32)
  img  = cv2.resize(img, (input_width, input_height))

  return img

def preprocess_default(iter, batch_size=QUANT_BATCH_SIZE, input_height=299, input_width=299, size_type=0, means=(104,107,123), scales=(1,1,1), normalize=False, **kwargs):
  images = []
  line = open(IMAGELIST).readlines()
  for index in range(0, batch_size):
    curline = line[iter * batch_size + index].strip()
    [calib_image_name, calib_label_id] = curline.split(' ')

    image = cv2.imread(IMAGEDIR + calib_image_name)
    if size_type == 0:
      image = central_crop(image, input_height, input_width)
    elif size_type == 1:
      image = resize(image, input_height, input_width)
    else:
      raise ValueError("Invalid size_type")
    image = means_subtraction(image, means)
    if scales != 1:
      image = scale_image(image, scales)
    if normalize != False:
      image = nomalize_image(image)
    image = convert_bgr_to_rgb(image)
    images.append(image)
  return {'input': images}

def preprocess_inception(iter, batch_size=QUANT_BATCH_SIZE, input_height=299, input_width=299, **kwarg):
  images = []
  labels = []
  line = open(IMAGELIST).readlines()
  for index in range(0, batch_size):
    curline = line[iter * batch_size + index].strip()
    [calib_image_name, calib_label_id] = curline.split(' ')

    image = cv2.imread(IMAGEDIR + calib_image_name)
    image = cv2.resize(image, (input_height, input_width))
    image = image/256.0
    image = image-0.5
    image = image*2
    images.append(image)
    labels.append(int(calib_label_id) + 1)
  # dict key below should be the name of graph inputs
  return {"input": images}



if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--model', default="", help='User must provide the network model file')
  parser.add_argument('--input_nodes', default="", help='User must provide the network input names [comma seperated with no spacing]')
  parser.add_argument('--output_nodes', default="", help='User must provide the network output names [comma seperated with no spacing]')
  parser.add_argument('--input_shapes', default="", help='User must provide the network input shapes [comma seperated with no spacing]')
  parser.add_argument('--input_means', default='104,107,123', help='User must provide the network input means [comma seperated with no spacing]')
  parser.add_argument('--scales', type=str, default="1,1,1", help="The scales of images per channel, comma separated. Images will be multiplied by scale for each channel.")
  parser.add_argument('--input_fn', default="default", help='User can provide the network preprocessing function')
  parser.add_argument('--output_dir', default="work", help='Optionally, save all generated outputs in specified folder')
  parser.add_argument('--quantize', action="store_true", default=False, help='In quantize mode, model will be Quantize')
  parser.add_argument('--validate_cpu', action="store_true", help='If validation_cpu is enabled, the model will be validated on cpu')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be partitioned, compiled, and ran on the FPGA, and the validation set examined')
  parser.add_argument('--c_input_nodes', default=None, help='Compiler input node names')
  parser.add_argument('--c_output_nodes', default=None, help='Compiler output node names')
  parser.add_argument('--image', default=None, help='User can provide an image to run')
  parser.add_argument('--numBatches', type=int, default=1, help='User must provide number of batches to run')
  args = dict2attr(parser.parse_args())

  if args.quantize:
    if os.path.isdir(args.output_dir):
      print('Cleaning model artifacts in {}'.format(os.path.abspath(args.output_dir)))
      filesToClean = [os.path.join(os.path.abspath(args.output_dir),f) for f in os.listdir(args.output_dir)]
      for f in filesToClean:
        os.remove(f)
    else:
      os.makedirs(args.output_dir)

    subprocess.call(['decent_q', 'inspect', '--input_frozen_graph', args.model])
    subprocess.call(['decent_q', 'quantize', '--input_frozen_graph', args.model,
                     '--input_nodes', args.input_nodes, '--output_nodes', args.output_nodes,
                     '--input_shapes', '{},{},{},{}'.format(*args.input_shapes), '--input_fn', args.input_fn,
                     '--method', '1', '--output_dir', args.output_dir,
                     '--image_dir', IMAGEDIR, '--image_list', IMAGELIST, '--means', '{},{},{}'.format(*args.input_means),
                     '--calib_iter', '1', '--batch_size', str(QUANT_BATCH_SIZE)])
    subprocess.call(['python', '-m', 'xfdnn.tools.compile.bin.xfdnn_compiler_tensorflow',
                     '-n', args.output_dir+'/deploy_model.pb', '-m', '9', '--dsp', '96',
                     '-g', args.output_dir+'/fix_info.txt'])

    print("Generated model artifacts in %s"%os.path.abspath(args["output_dir"]))
    for f in os.listdir(args["output_dir"]):
      print("  "+f)

  if args.input_fn == 'default':
    input_fn = preprocess_default
  else:
    module = __import__(args.input_fn.rsplit('.', 1)[0], fromlist=True)
    input_fn = getattr(module, args.input_fn.rsplit('.', 1)[1])

  if args.validate_cpu:
    eval_iters = 20
    eval_batch = 25

    tf.reset_default_graph()
    with open(args.model, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
      input_tensors = {node: sess.graph.get_operation_by_name(node).outputs[0] for node in make_list(args.input_nodes)}
      output_tensor = sess.graph.get_operation_by_name(args.output_nodes).outputs[0]

      top1_acc = 0
      top5_acc = 0
      progress = ProgressBar()
      line = open(IMAGELIST).readlines()
      for iter in progress(range(eval_iters)):
        correct_label = []
        for index in range(eval_batch):
          curline = line[iter * eval_batch + index].strip()
          [__, calib_label_id] = curline.split(' ')
          correct_label.append(int(calib_label_id) + 1)
        correct_label = np.array(correct_label)

        images = input_fn(iter, batch_size=eval_batch, input_height=args.input_shapes[1], input_width=args.input_shapes[2], size_type=0, means=args.input_means, scales=args.scales, normalize=False)
        predictions = sess.run(output_tensor, feed_dict={tensor: images[name] for name, tensor in input_tensors.items()})

        top1_prediction = np.argmax(predictions, axis=1)
        top5_prediction = np.argsort(predictions, axis=1)[:,-5:]
        top1_accuracy = sum(top1_prediction == correct_label)
        top5_accuracy = sum([label in top5_prediction for label in correct_label])
        top1_acc += top1_accuracy
        top5_acc += top5_accuracy
      total_samples = float(eval_iters*eval_batch)
      final_top1_acc = top1_acc/total_samples
      final_top5_acc = top5_acc/total_samples
      print ('top1_acc:{}, top5_acc:{}'.format(final_top1_acc,final_top5_acc))

  if args.validate:
    eval_iters = 100
    eval_batch = 1

    ### Partition and compile
    ## load default arguments
    FLAGS, unparsed = default_xdnn_arg_parser().parse_known_args([])

    rt = xdnnRT(FLAGS,
                networkfile=args.model,
                quant_cfgfile=args.output_dir+'/fix_info.txt',
                startnode=args.c_input_nodes,
                finalnode=args.c_output_nodes,
                xclbin=XCLBIN,
                device='FPGA',
                placeholdershape="{\'%s\': [%d,%d,%d,%d]}" % (args.input_nodes, eval_batch,args.input_shapes[1],args.input_shapes[2],args.input_shapes[3]),
                **get_default_compiler_args()
               )

    ### Accelerated execution
    ## load the accelerated graph
    graph = rt.load_partitioned_graph()

    ## run the tensorflow graph as usual (additional operations can be added to the graph)
    with tf.Session(graph=graph) as sess:
      input_tensors = {node: sess.graph.get_operation_by_name(node).outputs[0] for node in make_list(args.input_nodes)}
      output_tensor = sess.graph.get_operation_by_name(args.output_nodes).outputs[0]

      top1_acc = 0
      top5_acc = 0
      progress = ProgressBar()
      line = open(IMAGELIST).readlines()
      for iter in progress(range(eval_iters)):
        correct_label = []
        for index in range(eval_batch):
          curline = line[iter * eval_batch + index].strip()
          [__, calib_label_id] = curline.split(' ')
          correct_label.append(int(calib_label_id) + 1)
        correct_label = np.array(correct_label)

        images = input_fn(iter, batch_size=eval_batch, input_height=args.input_shapes[1], input_width=args.input_shapes[2], size_type=0, means=args.input_means, scales=args.scales, normalize=False)
        predictions = sess.run(output_tensor, feed_dict={tensor: images[name] for name, tensor in input_tensors.items()})

        top1_prediction = np.argmax(predictions, axis=1)
        top5_prediction = np.argsort(predictions, axis=1)[:,-5:]
        top1_accuracy = sum(top1_prediction == correct_label)
        top5_accuracy = sum([label in top5_prediction for label in correct_label])
        top1_acc += top1_accuracy
        top5_acc += top5_accuracy
      total_samples = float(eval_iters*eval_batch)
      final_top1_acc = top1_acc/total_samples
      final_top5_acc = top5_acc/total_samples
      print ('top1_acc:{}, top5_acc:{}'.format(final_top1_acc,final_top5_acc))
