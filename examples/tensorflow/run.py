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

import tensorflow as tf
import numpy as np
import cv2

# Bring in ml-suite Quantizer, Compiler, and Partitioner
from xfdnn.rt.xdnn_rt_tf import TFxdnnRT as xdnnRT
from xfdnn.rt.xdnn_util import dict2attr, make_list
from xfdnn.rt.xdnn_io import default_xdnn_arg_parser


# Environment Variables (obtained by running "source overlaybins/setup.sh")
IMAGEDIR   = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/"
IMAGELIST  = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val.txt"
LABELSLIST = "/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"
XCLBIN     = '/opt/ml-suite' + '/overlaybins/' + os.getenv('MLSUITE_PLATFORM','alveo-u200') + '/overlay_4.xclbin'

CALIB_BATCH_SIZE = 1

def calib_input(iter):
  images = []
  line = open(IMAGELIST).readlines()
  for index in range(0, CALIB_BATCH_SIZE):
    curline = line[iter * CALIB_BATCH_SIZE + index].strip()
    [calib_image_name, calib_label_id] = curline.split(' ')

    width_out = 128
    height_out = 224

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

    means = (123, 117, 104)
    image -= means

    images.append(image)
  return {"data": images}


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


def preprocess(image, means):
    input_height, input_width = 224, 224

    ## Image preprocessing using numpy
    img  = cv2.imread(image).astype(np.float32)
    img -= np.array(make_list(means)).reshape(-1,3).astype(np.float32)
    img  = cv2.resize(img, (input_width, input_height))

    return img




if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--model', default="", help='User must provide the network model file')
  parser.add_argument('--input_name', default="", help='User must provide the network input names [comma seperated with no spacing]')
  parser.add_argument('--output_name', default="", help='User must provide the network output names [comma seperated with no spacing]')
  parser.add_argument('--input_shapes', default="", help='User must provide the network input shapes [comma seperated with no spacing]')
  parser.add_argument('--input_means', default='104,107,123', help='User must provide the network input means [comma seperated with no spacing]')
  parser.add_argument('--output_dir', default="work", help='Optionally, save all generated outputs in specified folder')
  parser.add_argument('--quantize', action="store_true", default=False, help='In quantize mode, model will be Quantize')
  parser.add_argument('--q_calibIter', type=int, default=1, help='User can provide the number of iterations to run the quantization')
  parser.add_argument('--q_numBatches', type=int, default=1, help='User can provide the number of batches to run the quantization')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be partitioned, compiled, and ran on the FPGA, and the validation set examined')
  parser.add_argument('--c_input_name', default=None, help='Compiler input node names')
  parser.add_argument('--c_output_name', default=None, help='Compiler output node names')
  parser.add_argument('--image', default=None, help='User can provide an image to run')
  parser.add_argument('--numBatches', type=int, default=1, help='User must provide number of batches to run')
  args = dict2attr(parser.parse_args())

  CALIB_BATCH_SIZE = args.q_numBatches

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
                     '--input_nodes', args.input_name, '--output_nodes', args.output_name,
                     '--input_shapes', ','.join(args.input_shapes), '--input_fn', 'default',
                     '--method', '1', '--output_dir', args.output_dir,
                     '--image_dir', IMAGEDIR, '--image_list', IMAGELIST, '--means', '{},{},{}'.format(*args.input_means),
                     '--calib_iter', '1', '--batch_size', '10'])
    subprocess.call(['python', '-m', 'xfdnn.tools.compile.bin.xfdnn_compiler_tensorflow',
                     '-n', args.output_dir+'/deploy_model.pb', '-m', '9', '--dsp', '96',
                     '-g', args.output_dir+'/fix_info.txt'])

    print("Generated model artifacts in %s"%os.path.abspath(args["output_dir"]))
    for f in os.listdir(args["output_dir"]):
      print("  "+f)

  if args.validate:
    ### Partition and compile
    ## load default arguments
    FLAGS, unparsed = default_xdnn_arg_parser().parse_known_args([])

    rt = xdnnRT(FLAGS,
                networkfile=args.model,
                quant_cfgfile=args.output_dir+'/fix_info.txt',
                startnode=args.c_input_name,
                finalnode=args.c_output_name,
                xclbin=XCLBIN,
                device='FPGA',
                **get_default_compiler_args()
               )

    ### Accelerated execution
    ## load the accelerated graph
    graph = rt.load_partitioned_graph()

    ## run the tensorflow graph as usual (additional operations can be added to the graph)
    with tf.Session(graph=graph) as sess:
        input_tensor  = graph.get_operation_by_name(args.input_name).outputs[0]
        output_tensor = graph.get_operation_by_name(args.output_name).outputs[0]

        predictions = sess.run(output_tensor, feed_dict={input_tensor: [preprocess(args.image, args.input_means)]})

    labels = np.loadtxt(LABELSLIST, str, delimiter='\t')
    top_k = predictions[0].argsort()[:-6:-1]

    for l,p in zip(labels[top_k], predictions[0][top_k]):
        print (l," : ",p)
