##################################################
# Copyright 2018 Xilinx Inc.
##################################################
# The information disclosed to you hereunder (the "Materials") is provided solely for the selection and use of Xilinx products. To the
# maximum extent permitted by applicable law: (1) Materials are made available "AS IS" and with all faults, Xilinx hereby DISCLAIMS ALL
# WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF
# MERCHANTABILITY, NON-INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether in
# contract or tort, including negligence, or under any other theory of liability) for any loss or damage of any kind or nature related to,
# arising under, or in connection with, the Materials (including your use of the Materials), including for any direct, indirect, special,
# incidental, or consequential loss or damage (including loss of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was reasonably foreseeable or Xilinx had been advised of the
# possibility of the same. Xilinx assumes no obligation to correct any errors contained in the Materials or to notify you of updates to the
# Materials or to product specifications. You may not reproduce, modify, distribute, or publicly display the Materials without prior written
# consent. Certain products are subject to the terms and conditions of Xilinx's limited warranty, please refer to Xilinx's Terms of Sale which
# can be viewed at http://www.xilinx.com/legal.htm#tos; IP cores may be subject to warranty and support terms contained in a license
# issued to you by Xilinx. Xilinx products are not designed or intended to be fail-safe or for use in any application requiring fail-safe
# performance; you assume sole risk and liability for use of Xilinx products in such critical applications, please refer to Xilinx's Terms of
# Sale which can be viewed at http://www.xilinx.com/legal.htm#tos.
##################################################

import tensorflow as tf

import sys
import os
import argparse
import numpy as np
import os.path as osp


import tensorflow as tf
import network as net
from xfdnn_compiler_tensorflow import TFFrontend as xfdnnCompiler

import tensor_tools as tt

from tensorflow.python.platform import gfile

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  result = None
  with tf.Session() as sess :
    result = sess.run(normalized)
  return result


def display_results(dict_file, image_paths, probs) :
    
    # Get a list of ImageNet class labels
    with open(dict_file, 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
        
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        class_name = class_name[class_name.find(' ')+1:]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


def prepare_image_for_caffemodel(fname):
  image1 = tf.image.decode_jpeg(tf.read_file(fname), channels=3)
  image1 = tf.reverse(image1, axis=[-1]) # convert to BGR
  batch1out = tf.expand_dims(image1,0)
  resized1  = tf.image.resize_images(batch1out, [224, 224], tf.image.ResizeMethod.AREA)
  IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
  mean_image = tf.subtract(resized1, IMG_MEAN)
  return mean_image

def prepare_Inputs(sess, prepinps, feed_dict) :
    prep_nodes = {}
    for node_name in prepinps :
        tensor = sess.graph.get_tensor_by_name(node_name+':0')
        prep_nodes[node_name] = np.transpose(sess.run(tensor, feed_dict), [0,3,1,2])
    return prep_nodes

def TFlayerName2QuantizeKey(name):
  origName = name
  try:
    name = name.split("/", 1)[0]
    underscores = [i for i, ltr in enumerate(name) if ltr == '_']
    name_list = list(name)
    if len(underscores) <= 2:
      if "inception" in name:
        name_list[underscores[1]] = '/'
      else:
        name_list[underscores[0]] = '/'
    elif len(underscores) > 2:
      name_list[underscores[1]] = '/'
    name = ''.join(name_list)
  except:
    name = origName

  return name


def list_outputs_of_graph(graph_file) :
    graphdef = tf.GraphDef()
    with gfile.FastGFile(graph_file, 'rb') as f :
        graphdef.ParseFromString(f.read())
    inpset = set()
    for node in graphdef.node :
        for inp in node.input :
            inpset.add(inp)
    res = []
    for node in graphdef.node :
        if node.name not in inpset :
            res.append(node.name)
    return res

def list_inputs_of_graph(graph_file) :
    graphdef = tf.GraphDef()
    with gfile.FastGFile(graph_file, 'rb') as f :
        graphdef.ParseFromString(f.read())
    res = []
    for node in graphdef.node :
        if node.op == 'Placeholder' :
            res.append(node.name)
    return res

def main(args) :
    res = None
    #graph_file = '../tf/caffe_converted/googlenet_v1.pb'
    #graph_file = '../tf/caffe_converted/resnet_50.pb'
    #image = '/XLNX_DEV/Jupyter/imagenet/cropped_panda.jpg'
    #image = '/wrk/acceleration/users/aaronn/tf_retrain/flower_photos/roses/3873271620_1d9d314f01_n.jpg'
    #image = '/wrk/acceleration/users/aaronn/tf_retrain/flower_photos/roses/16229215579_e7dd808e9c.jpg'
    #image = '/wrk/hdstaff/satyakee/Downloads/rose.jpg'
    #image_data = tf.gfile.FastGFile(image, 'rb').read()
    #dict_file = '../../../../../examples/classification/synset_words.txt'                         
    #dict_file = '/wrk/acceleration/users/aaronn/tf_retrain/example_code/output_labels.txt'
    graph_file = args.networkfile
    final_node = args.finalnode
    image = args.test_img
    dict_file=args.labels
    if final_node == None :
        final_node = list_outputs_of_graph(graph_file)[0]      # assuming that there is only one output
    args.finalnode = final_node
    args.lasttensorbyname = final_node
    prepinps = list_inputs_of_graph(graph_file)
    print prepinps
    '''compiler = xfdnnCompiler(networkfile = graph_file,
                              weights = False,
                              strategy = args.strategy,
                              generatefile = args.generatefile,
                              dsp = args.dsp, # 28 -> 64, 56 - > 128  
                              memory=args.memory, # in MB 
                              ddr=args.ddr,
                              finalnode= final_node)'''
    compiler = xfdnnCompiler(args)

    graph, schedule, _ = compiler.compile()
    print 'Done compiling'
    with tf.Session() as sess :
        runCustom = True
        
        #prepinps = ['Placeholder']
        Net = net.net(graph, schedule, prepinps, final_node, runCustom)
        print 'CDBG:network created'

        if runCustom and args.doemu == "True":
          print 'CDBG:running hw emu'
          # Inception v1
          quantize_recipe = {
            "start": "module_apply_default/resnet_v2_50/conv1/Conv2D",
            "end": "module_apply_default/resnet_v2_50/pool5",
            "quantize": {"module_apply_default/resnet_v2_50/conv1/Conv2D":"module_apply_default/resnet_v2_50/conv1/Conv2D"},
            "unquantize": {"module_apply_default/resnet_v2_50/pool5":"module_apply_default/resnet_v2_50/pool5"},
            "name2key": TFlayerName2QuantizeKey
          }
          #quantize_recipe = {
          #  "start": "conv1_7x7_s2/Conv2D",
          #  "end": "conv1_7x7_s2/Conv2D",
          #  "quantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
          #  "unquantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
          #  "name2key": TFlayerName2QuantizeKey
          #}
          Net.quantize_schedule(quantize_recipe)

        #print Net.schedule
        #prep = prepare_Inputs(sess, prepinps, {'DecodeJpeg/contents:0':image_data})
        prep = {}
        prep[prepinps[0]] = np.transpose(read_tensor_from_image_file(image), [0,3,1,2])     # assuming that there is only one input
        res = Net.feed_forward(sess, prep, final_node)
        display_results(dict_file, [image],res)



if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=int, default=8)
    parser.add_argument("--generatefile", type=str, default='output.cmd')
    parser.add_argument("--dsp", type=int, default=28)
    parser.add_argument("--ddr", type=int, default=4096)
    parser.add_argument("--versionjson", type=str, default=None)
    parser.add_argument("--fromtensorflow", default=True)
    parser.add_argument("--weights", default=False)
    parser.add_argument("--strategy", type=str, default="all")
    parser.add_argument("--schedulefile", type=str, default='memschedule.txt')
    parser.add_argument("--pngfile", type=str, default='classify.png')
    parser.add_argument("--rankdir", type=str, default='BT')
    parser.add_argument("--finalnode", type=str, default=None)
    parser.add_argument("--doemu", type=str, default='False')
    parser.add_argument("--networkfile", type=str, default='/wrk/acceleration/users/aaronn/tf_retrain/example_code/retrained_graph.pb')
    parser.add_argument("--labels", type=str, default='/wrk/acceleration/users/aaronn/tf_retrain/example_code/output_labels.txt')
    parser.add_argument("--test_img", type=str, default="/wrk/hdstaff/satyakee/Downloads/rose.jpg")
    parser.add_argument("--placeholdershape", type=str, default=None)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--lasttensorbyname", type=str, default=None)
    args = parser.parse_args()
    main(args)


