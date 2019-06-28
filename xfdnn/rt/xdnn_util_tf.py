#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2019, Xilinx, Inc.
#

from collections import defaultdict, Iterable
from orderedset import OrderedSet

import numpy as np
import tensorflow as tf
from tensorflow.contrib.graph_editor import util as _util
from tensorflow.python.platform import gfile as _gfile
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_shape as _tensor_shape
from google.protobuf import text_format

from xfdnn.rt.xdnn_util import dict2attr, make_list








def strip_node_name(node_name):
  """Strips off ports and other decorations to get the underlying node name."""
  if node_name.startswith("^"):
    node_name = node_name[1:]
  return node_name.split(':')[0]



## expanding tf.NodeDef methods
def set_name(self, name):
  self.name = name

def set_shape(self, shape):
  if 'shape' in self.attr and len(self.attr['shape'].shape.dim) != len(shape):
    raise ValueError('dimension mismatch')
  TensorShapeProto = _tensor_shape.as_shape(shape).as_proto()
  self.attr['shape'].shape.CopyFrom(TensorShapeProto)

def get_shape(self):
  ret = []
  if 'shape' in self.attr:
    ret = [dim.size if dim.size>0 else None for dim in self.attr['shape'].shape.dim]
  elif '_output_shapes' in self.attr:
    ret = [dim.size if dim.size>0 else None for dim in
           self.attr['_output_shapes'].list.shape[0].dim]
  return ret

def get_dtype(self):
  if 'T' in self.attr:        ## if op
    dtype_enum = self.attr['T'].type
  elif 'dtype' in self.attr:  ## if tensors
    dtype_enum = self.attr['dtype'].type
  elif 'output_types' in self.attr:       ## if IteratorGetNext
    dtype_enum = self.attr['output_types'].list.type[0]
  else:
    raise AttributeError('NodeDef not supperted')
  return tf.DType(dtype_enum)




## expanding tf.GraphDef methods
def get_node_dict(self, outmap=False):
  node_dict = {}
  output_dict = defaultdict(lambda: {})
  for node in self.node:
    node_name = strip_node_name(node.name)

    if node_name in node_dict:
      raise KeyError('{} already exist in node dictionary'.format(node_name))
    else:
      node_dict[node_name] = node

    for input_index, input_name in enumerate(node.input):
      input_name = strip_node_name(input_name)
      output_dict[input_name][node_name] = input_index

  if outmap:
    return node_dict, output_dict
  return node_dict

def get_output_dict(self):
  output_dict = defaultdict(lambda: {})
  for node in self.node:
    node_name = strip_node_name(node.name)
    for input_index, input_name in enumerate(node.input):
      input_name = strip_node_name(input_name)
      output_dict[input_name][node_name] = input_index
  return output_dict

def get_node_index(self, node_name):
  for idx, node in enumerate(self.node):
    if node.name == node_name:
      return idx

def remove_nodes(self, node_names):
  for name in make_list(node_names):
    del self.node[self.get_node_index(name)]

def is_cyclic(self):
  output_dict = get_output_dict(self)

  def util(node_name, visited_set, cur_path):
    if node_name in cur_path:
      return True

    cur_path.add(node_name)
    if node_name not in visited_set:
      visited_set.add(node_name)
      for desc_name, __ in output_dict[node_name].items():
        desc_name = desc_name.split(':')[0]
        if util(desc_name, visited_set, cur_path):
          return True

    cur_path.pop()
    return False

  visited_set = set()
  cur_path    = OrderedSet()
  for node_name in get_node_dict(self).keys():
    if util(node_name, visited_set, cur_path):
      return True, cur_path
  return False, []

def all_cycles(self):
  cycles = []
  output_dict = get_output_dict(self)

  def util(node_name, visited_set, cur_path):
    if node_name in cur_path:
      cycles.append(list(cur_path)+[node_name])
      return

    cur_path.add(node_name)
    if node_name not in visited_set:
      visited_set.add(node_name)
      for desc_name, __ in output_dict[node_name].items():
        desc_name = desc_name.split(':')[0]
        util(desc_name, visited_set, cur_path)

    cur_path.pop()
    return

  visited_set = set()
  cur_path    = OrderedSet()
  for node_name in get_node_dict(self).keys():
    util(node_name, visited_set, cur_path)

  return cycles



## discover global sink nodes in a graph
def discover_sinknodes(graph_def):
  ## this alorithm work for the global graph (not subgraphs)
  ## find nodes that are not input to anyother node
  inp_set = set([inp.split(':')[0] for node in graph_def.node for inp in node.input])
  sinknodes = [node.name for node in graph_def.node if node.name not in inp_set]
  return sinknodes


def discover_sourcenodes(graph_def):
  return [node.name for node in graph_def.node if node.op == 'Placeholder']


def freeze_graph(sess, graph_def, remove_training_nodes=True, remove_redundant_nodes=True,
                 sinknodes_list=[], freeze_blacklist=[], freeze_whitelist=[], filename=None):
  ## freezing a graph_def by removing freeze_blacklist, training nodes(if remove_training_nodes), nodes not
  ## contributing in sinknode_list computation(if remove_redundant_nodes), while keeping
  ## freeze_blacklist (which includes the specified sinknodes_list)

  print('freeze model')

  sinknodes_list   = make_list(sinknodes_list)
  freeze_blacklist = make_list(freeze_blacklist)
  freeze_whitelist = make_list(freeze_whitelist)

  # if sess is not None:
  #   graph_def = sess.graph.as_graph_def(add_shapes=True)

  ## convert variables to constants for inference model
  if not sinknodes_list:
    sinknodes_list = discover_sinknodes(graph_def)

  if sess is not None:
    print('.... convert variables to constants')
    graph_def = tf.graph_util.convert_variables_to_constants(sess, graph_def, sinknodes_list)

  freeze_whitelist += sinknodes_list
  print('.... node count {}'.format(len(graph_def.node)))
  if remove_training_nodes:
    graph_def = tf.graph_util.remove_training_nodes(graph_def, protected_nodes=freeze_whitelist)
    print('.... node count after removing training nodes {}'.format(len(graph_def.node)))

  if remove_redundant_nodes:
    graph_def = tf.graph_util.extract_sub_graph(graph_def, sinknodes_list)
    print('.... node count after removing redundant nodes {}'.format(len(graph_def.node)))

  ## remove freeze_balcklist nodes
  # add summary nodes to freeze_balcklist
  freeze_blacklist += ['Summaries', 'MergeSummary']

  graph_def_frozen = tf.GraphDef()

  for node in graph_def.node:
    pass_cnd = np.array([blocked not in node.name for blocked in freeze_blacklist])
    if pass_cnd.all():
      graph_def_frozen.node.extend([node])
  print('.... node count after removing blacklisted nodes {}'.format(len(graph_def_frozen.node)))

  try:
    fValidGraph = True
    ## fill in all output shapes
    with tf.Graph().as_default() as temp_graph:
      tf.import_graph_def(graph_def_frozen, name='')
      graph_def_frozen2 = temp_graph.as_graph_def(add_shapes=True)
    graph_def_frozen = graph_def_frozen2

  except Exception as e:
    fValidGraph = False
    print(e)
    print(type(graph_def_frozen))
    #assert(False, "invalid graph_def")

  if filename is not None:
    print('save graph at {}'.format(filename))
    with tf.gfile.GFile(filename, "wb") as f:
      f.write(graph_def_frozen.SerializeToString())

  return graph_def_frozen, fValidGraph

_load_graph_args_keys = ['networkfile', 'loadmode', 'startnode', 'finalnode', 'inclusive', 'fixinputnames', 'placeholdershape', 'remove_training_nodes', 'remove_redundant_nodes', 'freeze_blacklist', 'freeze_whitelist', 'graph_savepath']

def load_graph(args, **kwargs):
    ## Loads the graph at args.networkfile. args can be either namedspace, namedtuple. or dictionary
    ## parameters:
    ##    networkfile:    path to the network file or folder
    ##    loadmode:       saving protocol of the network file
    ##    startnode:      list of source nodes of the graph. (optional. Defaults to all placehoders)
    ##    finalnode:      list of sink nodes of the graph. (optional. Defaults to all sinknodes)
    ##    inclusive:      include the starnodes. (optional. Defaults to True)
    ##    fixinputnames:  fix the input placeholder name. otherwise their names starts with geph__ and ends with a index. (optional. Defaults to True)
    ##    placeholdershape:         Dictionary mapping of placehoder shapes to new shapes
    ##    remove_training_nodes:    Limits the network to inference nodes if True (optional. Defaults to True)
    ##    remove_redundant_nodes:   Limits the network to nodes involved in computing the sinknodes (optional. Defaults to True)
    ##    freeze_blacklist:         list of nodes to keep in the graph (optional)
    ##    freeze_whitelist:         list of nodes to remove in the graph (optional)
    ##    graph_savepath:           path to save the updated graph (optional)
    ## return:
    ##    imported and modified graph_def
    ##    inputs to graph_def
    ##    outputs from graph_def

    args = dict2attr(args)
    args.update(kwargs)

    print('\n######### load_graph arguments #############')
    for key in _load_graph_args_keys:
      print('{}: {}'.format(key, args[key]))
    print('#############################################\n')

    sess             = None
    startnode        = make_list(args.startnode)
    finalnode        = make_list(args.finalnode)
    placeholdershape = args.get('placeholdershape', {})

    ## load graph_def
    if not args.networkfile:
      raise ValueError('networkfile is not specified!')

    graph_def = tf.GraphDef()
    with tf.Graph().as_default() as temp_graph:
      loadmode = args.get('loadmode', 'binary')

      if loadmode.lower() == 'checkpoint':
        sess = tf.Session(graph=temp_graph)
        tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], args.networkfile)
        graph_def = sess.graph.as_graph_def(add_shapes=True)

      elif loadmode.lower() == 'text':
        with open(args.networkfile) as f:
          graph_def = text_format.Parse(f.read(), tf.GraphDef())

      elif loadmode.lower() == 'binary':
        with _gfile.FastGFile(args.networkfile, 'rb') as f:
          graph_def.ParseFromString(f.read())

      else:
        raise ValueError('unsupported textmode parameter: \"{}\"'.format(loadmode))

      ## fill in all output shapes (Not necessary, will be performed in freeze_graph)
      # tf.import_graph_def(graph_def, name='')
      # graph_def = temp_graph.as_graph_def(add_shapes=True)

    inputs  = startnode if startnode else discover_sourcenodes(graph_def)
    outputs = finalnode if finalnode else discover_sinknodes(graph_def)

    if startnode or finalnode:
      graph_def = extract_subgraph(graph_def, outputs, inputs,
                                   inclusive=args.get('inclusive', True),
                                   fixinputnames=args.get('fixinputnames', True))

      inputs  = discover_sourcenodes(graph_def)
      outputs = discover_sinknodes(graph_def)

    if placeholdershape:
      node_dict = get_node_dict(graph_def)
      for name, shape in placeholdershape.items():
        print('change palceholder {} shape to {}'.format(name, shape))
        node = node_dict[name]
        set_shape(node, shape)

    graph_def, fValidGraph = freeze_graph(sess, graph_def,
                                          sinknodes_list=outputs,
                                          remove_training_nodes=args.get('remove_training_nodes', True),
                                          remove_redundant_nodes=args.get('remove_redundant_nodes', True),
                                          freeze_blacklist=args.get('freeze_blacklist', []),
                                          freeze_whitelist=args.get('freeze_whitelist', []),
                                          filename=args.graph_savepath)

    if sess:
      sess.close

    return graph_def, inputs, outputs, fValidGraph


## finding ops between a set of seed_ops and boundry_ops
def make_list_of_op(ops):
  if isinstance(ops, _ops.Graph):
    return ops.get_operations()
  else:
    if not isinstance(ops, Iterable):
      ops = [ops]
    if not ops:
      return []
    return [op for op in ops if isinstance(op, _ops.Operation)]

def get_within_boundary_ops(ops, seed_ops, boundary_ops=(), inclusive=False):
  ops          = make_list_of_op(ops)
  seed_ops     = make_list_of_op(seed_ops)
  boundary_ops = set(make_list_of_op(boundary_ops))
  res          = set(seed_ops)
  #if boundary_ops & res:
  #  raise ValueError("Boundary is intersecting with the seeds.")
  wave = set() if res == boundary_ops else set(seed_ops)
  while wave:
    new_wave = set()
    ops_io = [inp.op for op in wave for inp in op.inputs]
    for op in ops_io:
      if op in res:
        continue
      if op in boundary_ops:
        if inclusive:
          res.add(op)
      else:
        new_wave.add(op)
    res.update(new_wave)
    wave = new_wave
  return [op for op in ops if op in res]

def extract_subgraph(graph_def, outputs, inputs, inclusive=False, fixinputnames=False):
    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name='')

      seed_ops      = [graph.get_operation_by_name(output) for output in outputs]
      boundary_ops  = [graph.get_operation_by_name(input)  for input  in inputs ]
      ops = get_within_boundary_ops(graph, seed_ops, boundary_ops, inclusive)

      sgv = tf.contrib.graph_editor.make_view(ops)

      subgraph = tf.Graph()
      tf.contrib.graph_editor.copy(sgv, subgraph, reuse_dst_scope=True)

      subgraph_def = subgraph.as_graph_def()

      ######### FIXME: BREAKS for many files with same op names
      if fixinputnames:
        ## renaming the placeholders
        graph_inputs = [inp.op.name for inp in sgv.inputs]
        node_dict, \
        output_dict  = get_node_dict(subgraph_def, outmap=True)
        ph_names     = [name for name in node_dict.keys() if _util._DEFAULT_PLACEHOLDER_PREFIX in name]
        for name in ph_names:
          ######### FIXME: might cause issue in case input names has index
          subs_name = [rp for rp in graph_inputs if name.find(rp.split('/')[-1]) >= 0]

          if len(subs_name) == 0:
            raise RuntimeError('no mapping found')
          if len(subs_name) > 1:
            raise RuntimeError('many too one mapping')
          subs_name = subs_name[0]

          ## rename the node itself
          node = node_dict[name]
          node.name = subs_name

          ## rename associated consumer node inputs
          for cns_name, index in output_dict[name].items():
            cns_node = node_dict[cns_name]
            cns_node.input[index] = subs_name

    return subgraph_def
