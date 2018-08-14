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
from tensorflow.python.platform import gfile

import numpy as np
import sys
import os
import argparse
import os.path as osp


sys.path.append('~/xdfnn/MLsuite/xfdnn/tools/emu')
sys.path.append('~/xdfnn/MLsuite/xfdnn/tools')
sys.path.append('~/xdfnn/MLsuite/xfdnn/rt')
sys.path.append('~/xdfnn/MLsuite/xfdnn/tools/emu/examples')
sys.path.append('~/xdfnn/MLsuite/xfdnn')
sys.path.append('~/xdfnn/MLsuite') 


import re

import network_tf as net
#import tensor_tools as tt

'''

Extra code  required to  download the files

#def maybe_download_and_extract():
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    print(filepath)
    if not os.path.exists(filepath):
    
'''
class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self, label_lookup=None, uid_lookup=None):
        '''if not label_lookup_path:
            label_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')'''
        self.node_lookup = self.load(label_lookup, uid_lookup)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

        Args:
            label_lookup_path: string UID to integer node ID.
            uid_lookup_path: string UID to human-readable string.

        Returns:
            dict from integer node ID to human-readable string.
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in list(node_id_to_uid.items()):
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]

def prepare_image_for_caffemodel(fname):
  image1 = tf.image.decode_jpeg(tf.read_file(fname), channels=3)
  image1 = tf.reverse(image1, axis=[-1]) # convert to BGR
  batch1out = tf.expand_dims(image1,0)
  resized1  = tf.image.resize_images(batch1out, [224, 224], tf.image.ResizeMethod.AREA)
  IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
  mean_image = tf.subtract(resized1, IMG_MEAN)
  return mean_image

def display_results(label_file, num_top, image_path, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open(label_file, 'rb') as infile:
        class_labels = list(map(str.strip, infile.readlines()))

    # Display the results
    img_name = os.path.basename(image_path)

    print(img_name)

    res = probs.argsort()[0][-num_top:][::-1]
    for i in range(len(res)) :
      class_name = class_labels[res[i]]
      print("   %s %.4f" %  (class_name, probs[0][res[i]]))


def show_results(dict_file, image_paths, probs) :
    # Get a list of ImageNet class labels
    with open(dict_file, 'rb') as infile:
        class_labels = list(map(str.strip, infile.readlines()))
        
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    
    # Display the results
    print(('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence')))
    print(('-' * 70))
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        class_name = class_name[class_name.find(' ')+1:]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print(('{:20} {:30} {} %'.format(img_name, class_name, confidence)))


def preprocess_nodes(sess, nodes, feed_dict) :
  processed = {}
  for node in nodes :
    tensor = sess.graph.get_tensor_by_name(node+':0')
    processed[node] = sess.run(tensor, feed_dict)
  return processed

def run_resnet(args):
  graph_file = args.model
  img_dir = args.image_path
  image = args.image
  num_batch = args.num_batch
  b_size = args.batch
  if b_size < 1 :
    b_size = 1
  top = args.num_top
  run_satyaflow = False
  if args.custom == "True":
    run_satyaflow = True

  tf.reset_default_graph()
  with gfile.FastGFile(graph_file, 'rb') as f:
    graphdef = tf.GraphDef()
    graphdef.ParseFromString(f.read())
    tf.import_graph_def(graphdef, name='')
    print('Graph Imported')
  
  imgs = []
  pred = None
  with tf.Session() as sess :
    # Googlenet v3:
    # preprocess the data
    prep_nodes = []
    Net = net.net_tf(sess, prep_nodes, 'final_result', custom = run_satyaflow)
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    prep_inputs = {}
    pred = None
    if img_dir == None :
      prep_inputs = preprocess_nodes(sess, prep_nodes, {'DecodeJpeg/contents:0':image_data})
    else :
      imgs = []
      try :
        if img_dir[-1] != '/' :
            img_dir+='/'
        for f in os.listdir(img_dir) :
          if f[-4:] == '.jpg' :
            imgs.append(img_dir+f)
      except Exception as e :
        print(e)
      #print(imgs)
      if len(imgs) > 0 :
        bctr = 0
        ictr = 0
        for i in range(len(imgs)) :
          image_data = tf.gfile.FastGFile(imgs[i], 'rb').read()
          inp = preprocess_nodes(sess, prep_nodes, {'DecodeJpeg/contents:0':image_data})
          if ictr == 0 :
            prep_inputs = inp
          else :
            for k in list(inp.keys()) :
              prep_inputs[k] = np.concatenate((prep_inputs[k], inp[k]))
          ictr += 1
          if ictr == b_size or i == len(imgs)-1 :
            if bctr != 0 :
              pred = np.concatenate([pred, Net.feed_forward(sess, prep_inputs, 'final_result')])
            else :
              pred = Net.feed_forward(sess, prep_inputs, 'final_result')
            ictr = 0
            bctr += 1
            prep_inputs = {}
          if num_batch > 0 and bctr == num_batch :
            break
    #variables = Net.get_variables()

  # Googlenet v3
  node_lookup = NodeLookup(args.labels, args.uid)
  for x in range(len(pred)) :
    p = pred[x]
    res = p.argsort()[-top:][::-1]
    print('Image ',imgs[x]) 
    for i in range(len(res)) :
      human_string = node_lookup.id_to_string(res[i])
      score = p[res[i]]
      print('%s (score = %.5f)' % (human_string, score))

def run_googlenet_v3(args):
  graph_file = args.model
  img_dir = args.image_path
  image = args.image
  num_batch = args.num_batch
  b_size = args.batch
  if b_size < 1 :
    b_size = 1
  top = args.num_top
  run_satyaflow = False
  if args.custom == "True":
    run_satyaflow = True

  tf.reset_default_graph()
  with gfile.FastGFile(graph_file, 'rb') as f:
    graphdef = tf.GraphDef()
    graphdef.ParseFromString(f.read())
    tf.import_graph_def(graphdef, name='')
    print('Graph Imported')
  
  imgs = []
  pred = None
  with tf.Session() as sess :
    # Googlenet v3:
    # preprocess the data
    prep_nodes = ['Mul']
    Net = net.net_tf(sess, prep_nodes, 'softmax', custom = run_satyaflow)
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    prep_inputs = {}
    pred = None
    if img_dir == None :
      prep_inputs = preprocess_nodes(sess, prep_nodes, {'DecodeJpeg/contents:0':image_data})
    else :
      imgs = []
      try :
        if img_dir[-1] != '/' :
            img_dir+='/'
        for f in os.listdir(img_dir) :
          if f[-4:] == '.jpg' :
            imgs.append(img_dir+f)
      except Exception as e :
        print(e)
      #print(imgs)
      if len(imgs) > 0 :
        bctr = 0
        ictr = 0
        for i in range(len(imgs)) :
          image_data = tf.gfile.FastGFile(imgs[i], 'rb').read()
          inp = preprocess_nodes(sess, prep_nodes, {'DecodeJpeg/contents:0':image_data})
          if ictr == 0 :
            prep_inputs = inp
          else :
            for k in list(inp.keys()) :
              prep_inputs[k] = np.concatenate((prep_inputs[k], inp[k]))
          ictr += 1
          if ictr == b_size or i == len(imgs)-1 :
            if bctr != 0 :
              pred = np.concatenate([pred, Net.feed_forward(sess, prep_inputs, 'softmax')])
            else :
              pred = Net.feed_forward(sess, prep_inputs, 'softmax')
            ictr = 0
            bctr += 1
            prep_inputs = {}
          if num_batch > 0 and bctr == num_batch :
            break
    #variables = Net.get_variables()

  # Googlenet v3
  node_lookup = NodeLookup(args.labels, args.uid)
  for x in range(len(pred)) :
    p = pred[x]
    res = p.argsort()[-top:][::-1]
    print('Image ',imgs[x]) 
    for i in range(len(res)) :
      human_string = node_lookup.id_to_string(res[i])
      score = p[res[i]]
      print('%s (score = %.5f)' % (human_string, score))

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

def TFResnetlayerName2QuantizeKey(name):
  origName = name
  try:
    name = name.split("/", 1)[0]
    underscores = [i for i, ltr in enumerate(name) if ltr == '_']
    name_list = list(name)
    name = ''.join(name_list)
  except:
    name = origName

  #print "TFlayerName2QuantizeKey %s->%s" % (origName, name)

  return name

def run_converted_caffe(args):
  graph_file = args.model
  image = args.image
  top = args.num_top
  img_dir = args.image_path
  num_batch = args.num_batch
  b_size = args.batch
  if b_size < 1 :
    b_size = 1

  run_satyaflow = False
  if args.custom == "True":
    run_satyaflow = True

  tf.reset_default_graph()
  with gfile.FastGFile(graph_file, 'rb') as f:
    graphdef = tf.GraphDef()
    graphdef.ParseFromString(f.read())
    tf.import_graph_def(graphdef, name='')
    print('Graph Imported')
  
  pred = None

  with tf.Session() as sess :
    prepdata = {}
      
    prep_nodes = ['Placeholder']
    Net = net.net_tf(sess, prep_nodes, 'prob', custom=run_satyaflow)

    if run_satyaflow and args.doHwEmu == "True" and args.bunch == "False":
      # Inception v1
      quantize_recipe = {
        "start": "conv1_7x7_s2/Conv2D",
        "end": "inception_5b_output",
        "quantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
        "unquantize": {"inception_5b_output":"inception_5b/pool_proj"},
        "name2key": TFlayerName2QuantizeKey
      }
      # Resnet
      #quantize_recipe = {
      #  "start": "conv1/Conv2D",
      #  "end": "pool5",
      #  "quantize": {"conv1/Conv2D":"conv1"},
      #  "unquantize": {"pool5":"res5c_branch2c"},
      #  "name2key": TFResnetlayerName2QuantizeKey
      #}
      #quantize_recipe = {
      #  "start": "conv1_7x7_s2/Conv2D",
      #  "end": "conv1_7x7_s2/Conv2D",
      #  "quantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
      #  "unquantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
      #  "name2key": TFlayerName2QuantizeKey
      #}
      #quantize_recipe = {
      #  "start": "inception_5a_1x1/Conv2D",
      #  "end": "inception_5a_1x1/Conv2D",
      #  "quantize": {"inception_5a_1x1/Conv2D":"inception_5a/1x1"},
      #  "unquantize": {"inception_5a_1x1/Conv2D":"inception_5a/1x1"},
      #  "name2key": TFlayerName2QuantizeKey
      #}

      Net.quantize_schedule(quantize_recipe)

    if run_satyaflow and args.doHwEmu == "True" and args.bunch == "True":
      quantize_recipe = {
        "start": "conv2_3x3_reduce/Conv2D",
        "end": "pool5_7x7_s1",
        "quantize": {"conv2_3x3_reduce/Conv2D" : "conv2/3x3_reduce" },
        "unquantize": {"pool5_7x7_s1": "inception_5b/pool_proj"},
        "name2key": TFlayerName2QuantizeKey
      }

      Net.quantize_bunch_schedule(quantize_recipe)

    if run_satyaflow and args.FPGA == "True" and args.singleStep=="True":
      if args.xdnnv3 == "True":
#        xclbin = "../../../../../overlaybins/1525/my_system.xclbin"
        xclbin = "../../../../../overlaybins/1525/xdnnv3_3slr_5.1_1kernel_16b_Jun11th_d06111807_csrVer30.xclbin"
      else:
        xclbin = "../../../../../overlaybins/1525/xdnn_v2_32x28_4pe_16b_4mb_bank21.xclbin"
      Net.xdnnEnv.init_fpga(xclbin,args.xdnnv3)
        
      import conv_layer
      convLayerNames = []
      for i,ol in enumerate(Net.schedule):
        if isinstance(ol, conv_layer.conv_layer):
          convLayerNames.append(ol.output)

      # replace all conv layers with FPGA
      for i,convLayerName in enumerate(convLayerNames):
        if args.xdnnv3 == "True" and convLayerName == 'conv1_7x7_s2/Conv2D':
          continue
        else:
          fpga_recipe = {
           "startFPGA": convLayerName,
           "endFPGA":  convLayerName,
           "name2key": TFlayerName2QuantizeKey
             }
          Net.gen_fpga_schedule(fpga_recipe)


    if run_satyaflow and args.FPGA == "True" and args.singleStep=="False":
      if args.xdnnv3 == "True":
        xclbin = "../../../../../overlaybins/1525/d07071822_xdnn_3slr_5.1_1kernel_16b.xclbin"#xdnnv3_3slr_5.1_1kernel_16b_Jun11th_d06111807_csrVer30.xclbin"
      else:
        xclbin = "../../../../../overlaybins/1525/xdnn_v2_32x28_4pe_16b_4mb_bank21.xclbin"
      Net.xdnnEnv.init_fpga(xclbin,args.xdnnv3)
        
      # replace all conv layers with FPGA
      fpga_recipe = {
        "startFPGA": "conv2_3x3_reduce/Conv2D",
        "endFPGA": "pool5_7x7_s1",
        "name2key": TFlayerName2QuantizeKey
        }
      Net.gen_bunch_fpga_schedule(fpga_recipe)

    pred = None
    if img_dir == None :
      image_data_fn = prepare_image_for_caffemodel(image)
      prepdata['Placeholder'] = image_data_fn.eval()
      pred = Net.feed_forward(sess, prepdata, 'prob')
      show_results(args.labels, [image],pred)
    else :
      files = os.listdir(img_dir)
      bctr = 0
      ictr = 0
      for i in range(len(files)) :
        if files[i][-4:] == '.jpg' :
          image_data_fn = prepare_image_for_caffemodel(img_dir+'/'+files[i])
          if ictr == 0 :
            prepdata['Placeholder'] = image_data_fn.eval()
          else :
            prepdata['Placeholder'] = np.concatenate([prepdata['Placeholder'], image_data_fn.eval()])
          ictr+=1
        if ictr == b_size or (i==len(files) - 1 and ictr != 0):
          if bctr == 0 :
            pred = Net.feed_forward(sess, prepdata, 'prob')
          else :
            pred = np.concatenate([pred, Net.feed_forward(sess, prepdata, 'prob')])
          ictr = 0
          bctr += 1

      pred = Net.feed_forward(sess, prepdata, 'prob')
    #print Net.variables["conv1_7x7_s2/Conv2D"]
  #display_results(args.labels, args.num_top, image, pred)

  display_results(args.labels, args.num_top, image, pred)
  #print pred

def run_inference(args) :
  #run_googlenet_v3(args)
  run_converted_caffe(args)
  #run_resnet(args)
        
if __name__ == '__main__' :
  parser = argparse.ArgumentParser()
  model_path = "imagenet"
  parser.add_argument("--model", type=str, default=model_path+"/classify_image_graph_def.pb")
  parser.add_argument("--image", type=str, default=model_path+"/cropped_panda.jpg")
  parser.add_argument("--image_path", type=str, default=None)
  parser.add_argument("--labels", type=str, default=model_path+"/imagenet_2012_challenge_label_map_proto.pbtxt")
  parser.add_argument("--uid", type=str, default=model_path+"/imagenet_synset_to_human_label_map.txt")
  parser.add_argument("--num_top", type=int, default=5)
  parser.add_argument("--num_batch", type=int, default=-1)
  parser.add_argument("--batch", type=int, default=4)
  parser.add_argument("--custom", type=str, default="False")
  parser.add_argument("--xdnnv3",type=str,default="False")
  parser.add_argument("--layerName",type=str,default="False")
  parser.add_argument("--FPGA",type=str,default="False")
  parser.add_argument("--singleStep",type=str,default="True")
  parser.add_argument("--doHwEmu",type=str,default="False")
  parser.add_argument("--bunch",type=str,default="False")
  args = parser.parse_args()
  run_inference(args)

