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

# Bring in ml-suite Quantizer, Compiler, and Partitioner
from xfdnn.rt.xdnn_rt_tf import TFxdnnRT as xdnnRT
from xfdnn.rt.xdnn_util import dict2attr
from xfdnn.rt.xdnn_io import default_xdnn_arg_parser
from utils import input_fn, top5_accuracy


# Environment Variables (obtained by running "source overlaybins/setup.sh")
XCLBIN = '/opt/ml-suite' + '/overlaybins/' + os.getenv('MLSUITE_PLATFORM','alveo-u200') + '/overlay_4.xclbin'

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





if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--model', default="", help='User must provide the network model file')
  parser.add_argument('--input_nodes', default="", help='User must provide the network input names [comma seperated with no spacing]')
  parser.add_argument('--output_nodes', default="", help='User must provide the network output names [comma seperated with no spacing]')
  parser.add_argument('--input_shapes', default="", help='User must provide the network input shapes [comma seperated with no spacing]')
  parser.add_argument('--output_dir', default="work", help='Optionally, save all generated outputs in specified folder')
  parser.add_argument('--label_offset', default="0", help='Optionally, label offset of the dataset')
  parser.add_argument('--quantize', action="store_true", default=False, help='In quantize mode, model will be Quantize')
  parser.add_argument('--validate_cpu', action="store_true", help='If validation_cpu is enabled, the model will be validated on cpu')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be partitioned, compiled, and ran on the FPGA, and the validation set examined')
  parser.add_argument('--c_input_nodes', default=None, help='Compiler input node names')
  parser.add_argument('--c_output_nodes', default=None, help='Compiler output node names')
  args = dict2attr(parser.parse_args())


  if args.quantize:
    if os.path.isdir(args.output_dir):
      print('Cleaning model artifacts in {}'.format(os.path.abspath(args.output_dir)))
      filesToClean = [os.path.join(os.path.abspath(args.output_dir),f) for f in os.listdir(args.output_dir)]
      for f in filesToClean:
        os.remove(f)
    else:
      os.makedirs(args.output_dir)

    subprocess.call(['decent_q', 'inspect',
                     '--input_frozen_graph', args.model])
    subprocess.call(['decent_q', 'quantize',
                     '--input_frozen_graph', args.model,
                     '--input_nodes', args.input_nodes,
                     '--output_nodes', args.output_nodes,
                     '--input_shapes', '{},{},{},{}'.format(*args.input_shapes),
                     '--output_dir', args.output_dir,
                     '--input_fn', 'utils.input_fn',
                     '--method', '1',
                     '--calib_iter', '100'])
    subprocess.call(['python',
                     '-m', 'xfdnn.tools.compile.bin.xfdnn_compiler_tensorflow',
                     '-n', args.output_dir+'/deploy_model.pb',
                     '-m', '9',
                     '--dsp', '96',
                     '-g', args.output_dir+'/fix_info.txt'])

    print("Generated model artifacts in %s"%os.path.abspath(args["output_dir"]))
    for f in os.listdir(args["output_dir"]):
      print("  "+f)

  if args.validate_cpu:
    iter_cnt = 20
    batch_size = 25

    tf.reset_default_graph()
    with open(args.model, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      tf.import_graph_def(graph_def, name='')

    graph = tf.get_default_graph()

    top5_accuracy(graph, args.input_nodes, args.output_nodes, iter_cnt, batch_size, args.label_offset)

  if args.validate:
    iter_cnt = 100
    batch_size = 1

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
                placeholdershape="{\'%s\': [%d,%d,%d,%d]}" % (args.input_nodes, batch_size, args.input_shapes[1], args.input_shapes[2], args.input_shapes[3]),
                **get_default_compiler_args()
               )

    ### Accelerated execution
    ## load the accelerated graph
    graph = rt.load_partitioned_graph()

    top5_accuracy(graph, args.input_nodes, args.output_nodes, iter_cnt, batch_size, args.label_offset)
