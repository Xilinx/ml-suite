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

import sys
import os
import argparse

sys.path.insert(0, '/XLNX_DEV/DeepLearning/xilinx/xfdnn/python/xdlf')
sys.path.insert(0,'/XLNX_DEV/DeepLearning/xilinx/tools/network/')
sys.path.insert(0,'/XLNX_DEV/DeepLearning/xilinx/tools/optimizations/')
sys.path.insert(0,'/XLNX_DEV/DeepLearning/xilinx/tools/graph/')
sys.path.insert(0,'/XLNX_DEV/DeepLearning/xilinx/tools/codegeneration/')
sys.path.insert(0, '/XLNX_DEV/DeepLearning/xilinx/tools/memory/')

import network as net
import tensor_tools as tt

def maybe_download_and_extract():
	"""Download and extract model tar file."""
	dest_directory = FLAGS.model_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	print(filepath)
	if not os.path.exists(filepath):
    

class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

	def __init__(self, label_lookup_path=None, uid_lookup_path=None):
		if not label_lookup_path:
			label_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
		if not uid_lookup_path:
			uid_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
		self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

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

def run_inference(args) :
	graph_file = args.model
	image = args.image
	top = args.num_top
	tf.reset_default_graph()
	with gfile.FastGFile(graph_file, 'rb') as f:
		graphdef = tf.GraphDef()
		graphdef.ParseFromString(f.read())
		tf.import_graph_def(graphdef, name='')
		print('Graph Imported')
	
	image_data = tf.gfile.FastGFile(image, 'rb').read()
	pred = None
	with tf.Session() as sess :
		gg, schedule = tt.get_pydot_graph_tensor(sess.graph,"First",rankdir="BT")
		Net = net.net(gg, schedule, 'Mul', custom = True)
		pred = Net.feed_forward(sess, {'DecodeJpeg/contents:0':image_data}, 'softmax')
	res = pred.argsort()[-top:][::-1]
	for i in range(len(res)) :
		human_string = node_lookup.id_to_string(i)
		score = pred[i]
		print('%s (score = %.5f)' % (human_string, score))
		



if __main__() :
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, default="/XLNX_DEV/Jupyter/imagenet/classify_image_graph_def.pb")
	parser.add_argument("--image", type=str, default = "/XLNX_DEV/Jupyter/imagenet/cropped_panda.jpg")
	parser.add_argument("--num_top", type=int, default=3)
	args = parser.parse_args()
	run_inference(args)

for i in range(res.shape[1]) :
	if res[0][i] > .001 :
		print i, res[0][i]
