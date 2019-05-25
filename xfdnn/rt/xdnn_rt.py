#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import sys as _sys
import json
import re
from os import mkdir as _mkdir
from os.path import exists as _exists
from ast import literal_eval as l_eval
from collections import defaultdict, OrderedDict
from copy import deepcopy

import tensorflow as tf
import caffe
import numpy as np
from tensorflow.python.ops import script_ops as _script_ops

from xfdnn_compiler_tensorflow import TFFrontend
from xfdnn_compiler_caffe import CaffeFrontend
import xdnn_util
import xdnn_tf_util
from xdnn_opt import CPUTransform, HWEmuTransform, FPGATransform
import PyTurboJPEG



## global variables
global_fpga_device    = 'cpu:0'   ## TODO: replace with FPGA:0
global_pyfunc_counter = 0
save = None

######################################################
## tensorflow specific utility functions
######################################################
## expanding tf.NodeDef methods
tf.NodeDef.set_shape = xdnn_tf_util.set_shape
tf.NodeDef.get_shape = xdnn_tf_util.get_shape
tf.NodeDef.get_dtype = xdnn_tf_util.get_dtype

## expanding tf.GraphDef methods
tf.GraphDef.get_node_dict   = xdnn_tf_util.get_node_dict
tf.GraphDef.get_output_dict = xdnn_tf_util.get_output_dict
tf.GraphDef.is_cyclic       = xdnn_tf_util.is_cyclic
tf.GraphDef.all_cycles      = xdnn_tf_util.all_cycles




######################################################
## XFDNN classes
######################################################

class LayerPartition(object):
  def __init__(self, index=0, supported=False, time_to_layer_list=None, layerparameter_dict={},
               layeroutput_dict={}):
    self.fValid    = True
    self.index     = index
    self.supported = supported
    if time_to_layer_list:
      z            = zip(*time_to_layer_list)
      self.schedule, \
      self.names   = list(z[0]), list(z[1])
    else:
      self.schedule, \
      self.names   = [], []
    self.inputs,   \
    self.outputs,  \
    self.inputMap, \
    self.outputMap = self.partition_boundries(layerparameter_dict, layeroutput_dict)

  def size(self):
    return len(self.names)

  def partition_boundries(self, layerparameter_dict, layeroutput_dict):
    '''
    returns sourcenodes, sinknodes.
    sourcenodes are exclusive, i.e., they do not belong to the layer, except when the sourcenode
    does not have an input itself.
    sinknodes are inclusive, i.e., they are part of the layer.
    sourcenode_map are the last node that was collapsed into the sinknodes.
    sinknode_map are the last node that was collapsed into the sinknodes.
    '''
    partition_name_set = set(self.names)
    sourcenodes = []
    sinknodes   = []
    bottoms     = []
    for layer in self.names:
      inps = layerparameter_dict[layer].bottoms
      if inps is None:
        sourcenodes += [layer]
      else:
        bottoms += inps

      outs = layeroutput_dict[layer].keys()
      ## NOTE: not outs might cause issues
      if not outs or any([output not in partition_name_set for output in outs]):
        sinknodes += [layer]

    sourcenodes += [bottom for bottom in bottoms if bottom not in partition_name_set]
    # sinknodes    = [layer for layer in self.names if layer not in set(bottoms)]

    sourcenode_map = {}
    for sourcenode in sourcenodes:
      ## TODO: Check with @Paolo if we are combining previous nodes?
      ## TODO: Check with @Paolo with before layers are added in order
      extras_and_future = layerparameter_dict[sourcenode].extras_and_future
      if extras_and_future is None or sourcenode in partition_name_set:
        sourcenode_map[sourcenode] = sourcenode
      else:
        extras_names = [extra if isinstance(extra, basestring) else extra.name for extra in extras_and_future]
        sourcenode_map[sourcenode] = extras_names[-1]

    sinknode_map = {}
    for sinknode in sinknodes:
      extras_and_future = layerparameter_dict[sinknode].extras_and_future
      if extras_and_future is None:
        sinknode_map[sinknode] = sinknode
      else:
        extras_names = [extra if isinstance(extra, basestring) else extra.name for extra in extras_and_future]
        ## TODO: Check with #Paolo with future layers are added in order
        sinknode_map[sinknode] = extras_names[-1]

    return sourcenodes, sinknodes, sourcenode_map, sinknode_map

  def update(self, layerparameter_dict, layeroutput_dict, others):
    if not isinstance(others, list):
      others = [others]
    for other in others:
      other.fValid   = False
      self.schedule += other.schedule
      self.names    += other.names
    self.inputs,   \
    self.outputs,  \
    self.inputMap, \
    self.outputMap = self.partition_boundries(layerparameter_dict, layeroutput_dict)

  def forward_exec(self, inputs, outputs=None, preprocess=None, save=None):
    if not outputs:
      outputs = self.outputs
    else:
      for output in outputs:
        if isinstance(output, list):
          raise TypeError('outputs should be flattened list of name strings')

    if not isinstance(inputs, list):
      inputs = [inputs]

    # Add network input to variables list
    # self.variables[self.inputs[0]] = preprocess(inputs)
    for name, inp in zip(self.inputs, inputs):
      self.variables[name] = inp
    #import pdb; pdb.set_trace()
    for layer in self.layers:
      #print('Forward Exec: {}'.format(layer.name))
      layer_inputs = [self.variables[inp] for inp in layer.inputs]
      if layer_inputs:
        self.variables[layer.output] = layer.forward_exec( layer_inputs )
        print('Forward Exec: {} Dimensions: {}'.format(layer.name, self.variables[layer.name].shape))
      elif len(inputs) == 1:
        self.variables[layer.output] = inputs[0]
      else:
        raise RuntimeError('layer input mismatch')
    if save is not None:
      for var in self.variables.keys():
        np.save(save+'/'+'_'.join(var.split('/'))+'.npy', self.variables[var])
    return [self.variables[name] for name in outputs]

class xdnnRT(object):
    def __init__(self, compilerFunc, args, **kwargs):
        args = xdnn_util.dict2attr(args)
        args.update(kwargs)

        if args.networkfile is None and args.loadpickle is None:
          raise AttributeError('network argument is missing')

        self.fPartitioned  = args.get('fPartition', True)
        self.networkfile   = args.networkfile
        self.data_format   = args.data_format
        self.save2modeldir = args.save2modeldir   # whether to save to mo directory

        name_postfix       = args.get('name_postfix', '_frozen')
        self.picklefile    = self.file_path('.pickle', name_postfix=name_postfix)   # save pydotGraph and compilerJson
        #args.savepickle = self.picklefile
        args.loadpickle = self.picklefile
        args.forntendonly = args.get('frontendonly', False)

        ## run compiler
        print "DBG", args
        compiler = compilerFunc(args)
        #                        savepickle=self.picklefile,
        #                        loadpickle=self.picklefile,
        #                        frontendonly=args.get('frontendonly', False))

        self._graph, \
        self.inputs, \
        self.outputs, \
        pydotGraph, \
        compilerSchedule, \
        ssize, \
        compilerJson \
            = compiler.compile()

        if compilerJson is None:
          raise RuntimeError('Compiler failed to produce valid schedule')

        ## store compiler json
        with open(self.file_path('.json', name_postfix=name_postfix+'-xdlfCompiler'), "w") as f:
          json.dump(compilerJson, f, sort_keys=True, indent=4, separators=(',',': '))

        if (self._graph  is None or
            self.inputs  is None or
            self.outputs is None):
          ## in case loading from a pickled file
          self._graph, \
          self.inputs, \
          self.outputs \
              = self.load_graph(args)

        self.unspt_set      = set(compilerJson['unsupported']['list'].keys()) # build unsupported layer set
        layerparameter_dict = {}
        layeroutput_dict    = defaultdict(lambda: {})
        for node in pydotGraph.get_nodes():
          LayerParameter = node.get('LayerParameter')
          layer_name = LayerParameter.name
          layerparameter_dict[layer_name] = LayerParameter
          if 'blob' not in layer_name and LayerParameter.bottoms:       ## FIXME: hack to remove output blobs from output list
            for input_index, input_name in enumerate(LayerParameter.bottoms):
              layeroutput_dict[input_name][layer_name] = input_index

        self.layerparameter_dict = layerparameter_dict
        self.pydotGraph = pydotGraph
        self.compilerSchedule = compilerSchedule
        self.compilerJson = compilerJson
        #import pdb; pdb.set_trace()
        ## partitioning the graph




        if self.fPartitioned:
          ## Discover partitions based on compiler schedule
          self.spt_set, \
          self.layer_spt_dict, \
          self.graph_partitions, \
              = self.partitions_from_compiler_schedule(layerparameter_dict, layeroutput_dict,
                                                      compilerSchedule)
        else:
          self.spt_set, \
          self.layer_spt_dict, \
          self.graph_partitions, \
              = set([]), {}, []

        ## optimize partitions
        self.graph_partitions = self.refine_graph_partitions(self.graph_partitions)

        ## device transformation for supported layers
        self.device_transforms(args)

        self.rebuild_graph()


    def file_path(self, file_extension, name_prefix='', name_postfix=''):
        path_split = self.networkfile.rsplit('/', 1)
        savepath = path_split[0] if self.save2modeldir and len(path_split) > 1 else '.'
        networkname = path_split[-1].rsplit('.', 1)[0]
        return '/'.join([savepath, name_prefix + networkname + name_postfix + file_extension])

    def load_graph(self, args, **kwargs):
        pass

    def list_inputs_of_graph(self, pydotGraph):
        raise NotImplementedError('list_inputs_of_graph method not implemented for xdnnRT')

    def list_outputs_of_graph(self, pydotGraph):
        if hasattr(self, 'outputs') and self.outputs is not None:
          return self.outputs

        sinknodes = []
        for node in pydotGraph.get_nodes():
            param = node.get('LayerParameter')
            if param.tops is None :
                sinknodes += param.bottoms

        return sinknodes

    def partitions_from_compiler_schedule(self, layerparameter_dict, layeroutput_dict, compilerSchedule):
        print('\nPartition FPGA (un)supported layers')

        ## partition consecutive supported layers in zip(*list(partition.next()[1]))[1]
        print('.... is_supported, layer_index, layer_name')
        layer_spt_dict  = defaultdict(lambda: [0, []])    # NOTE: format: {grp_idx: (spt_flag, t2l_list)}
        spt_set         = []
        partition_idx   = -1
        prev_spt        = True if compilerSchedule.time_to_layer[0][0] in self.unspt_set else False
        for time, layers in compilerSchedule.time_to_layer.items():
          for layer in layers:
            spt = False if layer in self.unspt_set else True
            if spt ^ prev_spt:
              partition_idx += 1
              layer_spt_dict[partition_idx][0] = spt
            layer_spt_dict[partition_idx][1] += [(time, layer)]
            prev_spt = spt

            ## build supported layer set
            if spt:
              spt_set += [layer]
              extras_and_future = layerparameter_dict[layer].extras_and_future
              if extras_and_future is not None:
                extras_names = [extra if isinstance(extra, basestring) else extra.name for extra in extras_and_future]
                spt_set += [name for name in extras_names if name not in self.unspt_set]

            print('.... {0:5s}, {1:3d}, {2:s}'.format(str(spt), time, layer))

        graph_partitions = [LayerPartition(grp_idx, spt, t2l_list, layerparameter_dict,
                                           layeroutput_dict) for grp_idx, (spt, t2l_list) in
                            layer_spt_dict.items()]

        return set(spt_set), layer_spt_dict, graph_partitions


    def refine_graph_partitions(self, graph_partitions):
        def partitions_connectivity(graph_partitions):
            # print('partition connectivity:')
            connectivity = defaultdict(lambda: defaultdict(list))
            reverse_connectivity = defaultdict(lambda: defaultdict(list))
            for index, partition in enumerate(graph_partitions):
              other_partitions = list(graph_partitions)
              other_partitions.pop(index)
              for other in other_partitions:
                temp_set = set(partition.inputs).intersection(set(other.names))
                if temp_set:
                  for name in temp_set:
                    reverse_connectivity[partition.index][name].append(other.index)

                temp_set = set(other.inputs).intersection(set(partition.names))
                if temp_set:
                  for name in temp_set:
                    connectivity[partition.index][name].append(other.index)

              # print('partition ({:3d}, {:5s}) --> {}'.format(partition.index, str(partition.supported),
              #                                                [(graph_partitions[index].index,
              #                                                  graph_partitions[index].supported) for
              #                                                 index_list in
              #                                                 connectivity[partition.index].values()
              #                                                 for index in index_list]))
              # print('partition ({:3d}, {:5s}) <-- {}'.format(partition.index, str(partition.supported),
              #                                                [(graph_partitions[index].index,
              #                                                  graph_partitions[index].supported) for
              #                                                 index_list in
              #                                                 reverse_connectivity[partition.index].values()
              #                                                 for index in index_list]))
            return connectivity, reverse_connectivity

        def topological_ordering(connectivity, reverse_connectivity, forward=True, subset_view=None,
                                 seed=None):
            connectivity_map = connectivity if forward else reverse_connectivity
            vertex_support = subset_view if subset_view else connectivity_map.keys()
            vertex_seed    = seed if seed else vertex_support

            def helper(vertex, ordering):
              if vertex not in vertex_support or visited[vertex] == 1:   ## processing vertex is done
                return
              elif visited[vertex] == 2: ## processing vertex is incomplete
                raise RuntimeError('graph is not an DAG')
              visited[vertex] = 2
              for output_name, v_list in connectivity_map[vertex].items():
                for v in v_list:
                  helper(v, ordering)
              visited[vertex] = 1
              ordering.append(vertex)

            visited = {v: 0 for v in vertex_support}   ## 0 if not visited, 1 if done, 2 if inprocess
            ordering = []
            for v in vertex_seed:
              if not visited[v]:
                helper(v, ordering)
            return ordering

        def topological_level(ordering, connectivity):
            ordering   = list(ordering)
            vertex_cnt = len(ordering)
            levels     = {v: vertex_cnt for v in ordering}
            vertex     = ordering.pop()
            levels[vertex] = 0
            while ordering:
              for output_name, v_list in connectivity[vertex].items():
                for v in v_list:
                  levels[v] = min(levels[v], levels[vertex]+1)
              vertex = ordering.pop()

            level2vertex = defaultdict(list)
            for v, l in levels.items():
              level2vertex[l].append(v)

            return levels, level2vertex

        def merge_partitions(graph_partitions, matchings):
            for src_idx, dst_idxs in matchings:
              src_partition = graph_partitions[src_idx]
              dst_partitions = [graph_partitions[dst_idx] for dst_idx in dst_idxs]
              src_partition.update(layerparameter_dict, layeroutput_dict, dst_partitions)

            graph_partitions = [partition for partition in graph_partitions if partition.fValid]

            ## fix the indices
            for idx, partition in enumerate(graph_partitions):
              partition.index = idx

            return graph_partitions

        def acyclic_matching(graph_partitions):
            def is_contraction_valid(connectivity, reverse_connectivity, ordering, src, dst):
                def affected_set(ordering, src, dst):
                    end_idx   = ordering.index(src) + 1
                    start_idx = ordering.index(dst)
                    return set(ordering[start_idx:end_idx])

                af_set = affected_set(ordering, src, dst)

                af_forward = af_set.copy(); af_forward.remove(dst)
                fDFS = topological_ordering(connectivity, reverse_connectivity, forward=True,
                                            subset_view=af_forward, seed=[src])

                af_backward = af_set.copy(); af_backward.remove(src)
                bDFS = topological_ordering(connectivity, reverse_connectivity, forward=False,
                                            subset_view=af_backward, seed=[dst])

                if set(fDFS).intersection(set(bDFS)):
                  return False
                print('contract src {} <-- dst {}'.format(src, dst))
                return True

            def summarize_matchings(matching):
                def helper(src, dst_list):
                    for dst in dst_list:
                      if dst in matching:
                        helper(dst, matching[dst])
                        matching[src] += matching[dst]
                        matching.pop(dst)

                for src in sorted(matching.keys()):
                  helper(src, matching[src])

            connectivity, \
            reverse_connectivity = partitions_connectivity(graph_partitions)

            topOrder = topological_ordering(connectivity, reverse_connectivity)

            matching = xdnn_util.UnionFind(len(topOrder))
            for src_v in topOrder[::-1]:
              src_support = graph_partitions[src_v].supported
              for name, v_list in connectivity[src_v].items():
                for dst_v in v_list:
                  dst_support = graph_partitions[dst_v].supported
                  if src_support == dst_support:
                    if is_contraction_valid(connectivity, reverse_connectivity, topOrder, src_v, dst_v):
                      matching.union(src_v, dst_v)
            merge_matches = matching.components()

            graph_partitions = merge_partitions(graph_partitions, list(merge_matches.items()))
            return graph_partitions




        #############################
        ## function's body
        #############################
        print('\nRefine Graph Partitions')

        graph_partitions = acyclic_matching(graph_partitions)

        ## FIX: for UBER
        #matchings = [(0, [10, 12, 14]), (2, [4, 6, 8]), (17, [3, 5, 7, 9, 11, 13, 15])]
        #graph_partitions = merge_partitions(graph_partitions, matchings)

        self.connectivity, \
        self.reverse_connectivity = partitions_connectivity(graph_partitions)

        #for partition in graph_partitions:
        #  if ('ssd_300_vgg/conv6/SpaceToBatchND' in partition.inputs or
        #      'ssd_300_vgg/block4_box/conv_cls/Conv2D' in partition.names):   ## FIXME: Hack to Support SSD. Compiler should handle this case
        #    partition.supported = False

          ## FIXME: Heuristic. Don't use fpga for small partitions
          #if partition.size() < 4:
          #  partition.supported = False

        return graph_partitions

    def device_transforms(self, args):
        raise NotImplementedError('device_transforms method not implemented for xdnnRT')

    def rebuild_graph(self):
        raise NotImplementedError('rebuild_graph method not implemented for xdnnRT')

    def preprocess(self,  inputs):
        raise NotImplementedError('preprocess method not implemented for xdnnRT')

    def batch_classify(self, img_list, batch, preprocess):
        bctr = 0
        ictr = 0
        pred = None
        prepdata = {}
        prep = self.inputs[0]
        ctr = 0
        pred = {}
        #import pdb; pdb.set_trace()
        while ctr < len(img_list):
            ctrmax = min(ctr+batch, len(img_list))
            print("processing", img_list[ctr:ctrmax])
            res = self.forward_exec(img_list[ctr:ctrmax], preprocess = preprocess)
            for k, v in res.items() :
                pred.setdefault(k, []).append(v)
            ctr = ctrmax
        if len(pred) == 0: return {}
        for k, v in pred.items() :
            if len(pred[k]) == 1 :
                pred[k] = v[0]
            else :
                pred[k] = np.concatenate(v)
        return pred



class CaffexdnnRT(xdnnRT):
    def __init__ (self, args, **kwargs):
        super(CaffexdnnRT, self).__init__(CaffeFrontend, args, **kwargs)

    def load_graph(self, args, **kwargs):
        graph = caffe.Net(args.networkfile, args.weights, caffe.TEST)
        self.save = args.save
        if self.save and not _exists(self.save):
          _mkdir(self.save)

        inputs  = self.inputs if self.inputs else self.list_inputs_of_graph(graph)
        outputs = self.outputs if self.outputs else self.list_outputs_of_graph(graph)
        return graph, inputs, outputs

    def list_inputs_of_graph(self, graph):
        res = []
        for name, layer in zip(graph._layer_names, graph.layers) :
          print name, layer.type
          if layer.type in ['Input']:
            res.append(name)
        return res

    def list_outputs_of_graph(self, graph):
        return ['prob']
        raise NotImplementedError('''list_outputs_of_graph method not implemented for CaffexdnnRT.
                                  (REMOVE *.pickle TO FIX THIS ISSUE FOR NOW!!!!)''')

    def extract_subgraph(self, outputs, inputs, inclusive=False, filename=None):
        pass

    def device_transforms(self, args):
        print "DEVICE",args.device
        for partition in self.graph_partitions:
          time_to_layer_list = zip(partition.schedule, partition.names)
          print partition.supported
          if partition.supported:
            if args.device == "CPU":
              opt = CPUTransform(time_to_layer_list, self.layerparameter_dict, args, self._graph)
            elif args.device == "HWEmu":
              opt = HWEmuTransform(time_to_layer_list, self.layerparameter_dict, args, self._graph)
              #raise RuntimeError('not implemented yet')
              #opt = HWEmuTransform(partition.inputs, pydotGraph, compilerSchedule, args)
            elif args.device == "FPGA":
                if not args.fpga_recipe:
                  args.fpga_recipe = {'start': [time_to_layer_list[0][1]], 'end': partition.outputs}
                if args.xclbin:
                  opt = FPGATransform(time_to_layer_list, self.layerparameter_dict, self.compilerJson, args, self._graph)
                else:
                  raise AttributeError("Must specify path to xclbin when device = FPGA")
            else:
              raise AttributeError("Unsupported device type", args.device)
          else:
            ## default back to CPU implementation
            opt = CPUTransform(time_to_layer_list, self.layerparameter_dict, args, self._graph)

          #variables hold the inputs/consts of graph
          partition.layers    = opt.getLayers()
          partition.variables = opt.variables
          for l in partition.layers:
            l.setup()

    def rebuild_graph(self):
        pass

    def forward_exec(self, inputs, outputs=None, preprocess=None, **kwargs):
        if not outputs:
          outputs = self.outputs
        else:
          for output in outputs:
            if isinstance(output, list):
              raise TypeError('outputs should be flattened list of name strings')
        print outputs
        if not preprocess:
          preprocess = self.preprocess
        res = {}
        data = preprocess(inputs, **kwargs)
        for partition in self.graph_partitions:
          data = partition.forward_exec(data, save=self.save)

          for out_name, out_val in zip(partition.outputs, data):
            if out_name in outputs:
              res.update({out_name: out_val})

        if self.save is not None:
          with open(self.save+'/layers.txt', 'w') as f:
            for partition in self.graph_partitions:
              for name in partition.names:
                f.write('_'.join(name.split('/'))+'\n')
          self.save = None
        return res



class TFxdnnRT(xdnnRT):
    def __init__ (self, args, **kwargs):
        super(TFxdnnRT, self).__init__(TFFrontend, args, **kwargs)

    def load_graph(self, args, **kwargs):
        return xdnn_tf_util.load_graph(args, **kwargs)

    def list_inputs_of_graph(self, graph_def):
        return xdnn_tf_util.discover_sourcenodes(graph_def)

    def list_outputs_of_graph(self, graph_def):
        if hasattr(self, 'outputs') and self.outputs is not None:
          return self.outputs
        return xdnn_tf_util.discover_sinknodes(graph_def)

    def extract_subgraph(self, outputs, inputs, inclusive=False, filename=None, session=None):
        destgraph_def = xdnn_tf_util.extract_subgraph(self._graph, outputs, inputs, inclusive)
        return xdnn_tf_util.freeze_graph(session, destgraph_def, sinknodes_list=outputs,
                                         filename=filename, freeze_blacklist=[],
                                         freeze_whitelist=[])

    def device_transforms(self, args):
        TFs = []
        for partition in self.graph_partitions:
          ## extract supported subgraph
          print('Transorm partition_index \"{}\"'.format(partition.index))
          print('inputs:  {}'.format(partition.inputMap.keys()))
          print('actual:  {}'.format(partition.inputMap.values()))
          print('outputs: {}'.format(partition.outputMap.keys()))
          print('actual:  {}'.format(partition.outputMap.values()))

          if partition.supported:
            post_fix = '-partition#{:02d}'.format(partition.index)
            filename = self.file_path('.pb', name_postfix=post_fix)

            TFs.append(TFxdnnRT(args,
                                startnode=partition.inputMap.values(),
                                finalnode=partition.outputMap.values(),
                                inclusive=False,
                                fixinputnames=True,
                                name_postfix=post_fix,
                                weights=filename,
                                graph_savepath=filename,
                                fPartition=False))

            time_to_layer_list = []
            for time, layers in TFs[-1].compilerSchedule.time_to_layer.items():
              for layer in layers:
                time_to_layer_list.append((time, layer))

            ## TODO: hack to get rid of input placeholder
            # time_to_layer_list = [tol for tol in time_to_layer_list if not tol[1].startswith('geph__')]
            time_to_layer_list = [tol for tol in time_to_layer_list if tol[1] not in set(TFs[-1].inputs)]

            if args.device.lower() == "cpu":
              opt = CPUTransform(time_to_layer_list, TFs[-1].layerparameter_dict, args, TFs[-1]._graph)
            elif args.device.lower() == "hwemu":
              opt = HWEmuTransform(partition.inputs, TFs[-1].pydotGraph, TFs[-1].compilerSchedule, args)
            elif args.device.lower() == "fpga":
              if not args.fpga_recipe:
                args.fpga_recipe = {'start': [time_to_layer_list[0][1]], 'end': partition.outputs}
              if args.xclbin:
                opt = FPGATransform(time_to_layer_list, TFs[-1].layerparameter_dict, TFs[-1].compilerJson, args,
                                    TFs[-1]._graph)
              else:
                raise AttributeError("Must specify path to xclbin when device = FPGA")
            else:
              raise AttributeError("Unsupported device type", args.device)

            #variables hold the inputs/consts of graph
            partition.layers    = opt.getLayers()
            partition.variables = opt.variables
            for l in partition.layers:
              l.setup()

          else:
            partition.layers    = list(partition.names)
            partition.variables = {}

    def rebuild_graph(self):
        if len(self.graph_partitions) > 0:
          self.rebuild_graph_def()

    def rebuild_graph_def(self):
        #############################
        ## helper function
        #############################
        def discover_consumer_nodes(root_name, partition_node_set, graph_output_dict):
            dst_dict = {}
            consumer_nodes = graph_output_dict[root_name]
            for consumer_name, index in consumer_nodes.items():
              if consumer_name not in partition_node_set:
                dst_dict[consumer_name] = index
            return dst_dict

        #def discover_consumer_nodes((root_name, index), partition_outputs, dst_dict):
        #    ## FIXME: This doesn't work if the supported partition is not maximal
        #    if root_name in self.spt_set:
        #      if index != -1 and root_name in partition_outputs:
        #        return
        #      consumer_nodes = graph_output_dict[root_name]
        #      for consumer_item in consumer_nodes.items():
        #        discover_consumer_nodes(consumer_item, partition_outputs, dst_dict)
        #    else:
        #      dst_dict[root_name] = index

        #############################
        ## helper function
        #############################
        def insert_fpga_pynode(partition, graph_node_dict, graph_output_dict):
            global global_pyfunc_counter

            input_partitions    = self.reverse_connectivity[partition.index]
            consumer_partitions = self.connectivity[partition.index]

            input_tensors           = []
            placeholder_replace_map = {}

            with tf.Graph().as_default() as fpga_subgraph:
              for inp in partition.inputs:
                ## NOTE: index of partition in self.graph_partitions must match partition.index
                input_partition = input_partitions[inp]
                if len(input_partition) > 1:
                  raise RuntimeError('input {} cannot belong to multiple partitions! {}'.format(inp, input_partition))
                if not self.graph_partitions[input_partition[0]].supported:
                  inp = partition.inputMap[inp]

                ## NOTE: inp shouldn't have ":" except for multiple outputs of py_func
                inp = inp.split(':')[0]

                ## assuming all py_func inputs are active tensors and FPGA IP data_format is 'NCHW'
                input_nodedef = graph_node_dict[inp]
                input_tensor  = tf.placeholder(input_nodedef.get_dtype(),
                                               shape=input_nodedef.get_shape(), name=inp)
                input_tensors += [input_tensor]
                placeholder_replace_map[input_tensor.op.node_def.name] = inp

              with fpga_subgraph.name_scope(xdnn_util.Trie(partition.names).lcs()):
                with fpga_subgraph.name_scope('fpga_func_{}'.format(global_pyfunc_counter)):
                  output_dtypes = [graph_node_dict[output].get_dtype() for output in partition.outputs]

                  if self.data_format != 'NCHW':
                    with fpga_subgraph.name_scope('fpga_preproc'):
                      fpga_input_tensors = [tf.transpose(input_tensor, [0, 3, 1, 2]) for input_tensor in
                                            input_tensors]
                  else:
                    fpga_input_tensors = input_tensors

                  #with tf.device(global_fpga_device):
                  fpga_output_tensors = tf.py_func(partition.forward_exec, fpga_input_tensors,
                                                   output_dtypes, stateful=False)
                  #print('graph py_func tokens: {}'.format(_script_ops._py_funcs._funcs.keys()))
                  if self.data_format != 'NCHW':
                    with fpga_subgraph.name_scope('fpga_postproc'):
                      output_tensors = [tf.transpose(tensor, [0, 2, 3, 1]) for tensor in fpga_output_tensors]
                  else:
                    output_tensors = fpga_output_tensors
                  #print('graph py_func tokens: {}'.format(_script_ops._py_funcs._funcs.keys()))
                  global_pyfunc_counter += 1

            ## maintain an ordered dictionary of pynodes for loading the graph
            #self.fpga_pynode_dict[fpga_output_tensors[0].op.node_def.attr['token'].s] = partition
            self.fpga_pynode_dict[fpga_output_tensors[0].op.name] = (partition, [inp.name for inp in
                                                                             fpga_input_tensors],
                                                                     list(output_dtypes))

            fpga_subgraph_def    = fpga_subgraph.as_graph_def(add_shapes=True)
            fpga_node_map        = fpga_subgraph_def.get_node_dict()
            fpga_output_node_map = fpga_subgraph_def.get_output_dict()

            fpga_nodes = []
            for fpga_node_name, fpga_node in fpga_node_map.items():
              if fpga_node.op == 'Placeholder':
                # replace palceholder with original inputs
                placeholder_consumers = fpga_output_node_map[fpga_node_name]
                for placeholder_consumer_name, input_index in placeholder_consumers.items():
                  placeholder_consumer_node = fpga_node_map[placeholder_consumer_name]
                  del placeholder_consumer_node.input[input_index]
                  placeholder_consumer_node.input.insert(input_index, placeholder_replace_map[fpga_node_name])
              else:
                # All nodes except dummy placeholders are to be copied to the main graph
                fpga_nodes += [fpga_node]

            # connect input of fpga_consumer_nodes to output from py_function
            for i, outp in enumerate(partition.outputs):
              #output_consumer_nodes = {}
              #discover_consumer_nodes((outp, -1), set(partition.outputs), output_consumer_nodes)
              output_consumer_nodes = discover_consumer_nodes(partition.outputMap[outp],
                                                              set(partition.names), graph_output_dict)

              for consumerPart in consumer_partitions[outp]:
                consumerPart = self.graph_partitions[consumerPart]
                #print(outp, output_consumer_nodes, consumerPart.index, consumerPart.supported)
                if consumerPart.supported:
                  temp = self.reverse_connectivity[consumerPart.index]
                  temp[output_tensors[i].name] = temp[outp]
                  temp.pop(outp)
                  consumerPart.inputs = [name if name != outp else output_tensors[i].name for name in consumerPart.inputs]
                  consumerPart.inputMap[output_tensors[i].name] = consumerPart.inputMap[outp]

              for output_consumer_node_name, input_index in output_consumer_nodes.items():
                output_consumer_node = graph_node_dict[output_consumer_node_name]
                del output_consumer_node.input[input_index]
                output_consumer_node.input.insert(input_index, output_tensors[i].name)
            return fpga_nodes

        #############################
        ## function's body
        #############################
        graph_def = deepcopy(self._graph)

        self.fpga_pynode_dict = OrderedDict()

        for t, partition in enumerate(self.graph_partitions):
          if partition.supported:
            graph_node_dict   = graph_def.get_node_dict()
            graph_output_dict = graph_def.get_output_dict()

            graph_def.node.extend(insert_fpga_pynode(partition, graph_node_dict, graph_output_dict))

        # check graph integrity
        isCyclic, cycle = graph_def.is_cyclic()
        if isCyclic:
          raise RuntimeError('Graph partitioning resulted in cyclic graph; {}'.format(graph_def.all_cycles()))

        ## To make sure the modified graph_def is valid
        #with tf.Graph().as_default():
        #  tf.import_graph_def(graph_def, name='')

        # cleanup graph (at this point we do not have variables in graph so no initialization (sess) is needed)
        self.graph_def = xdnn_tf_util.freeze_graph(None, graph_def, sinknodes_list=self.outputs,
                                                   filename=self.file_path('.pb',
                                                                           name_postfix='_fpga'),
                                                   freeze_blacklist=[], freeze_whitelist=[])


    def feed_forward_uber(self, inputs, outputs=None, preprocess=None, **kwargs):
        if not outputs:
          outputs = self.outputs
        if not preprocess:
          preprocess = self.preprocess
        if not isinstance(inputs, list):
          inputs = [inputs]

        config   = kwargs['config'] if 'config' in kwargs else None
        num_iter = kwargs['num_iter'] if 'num_iter' in kwargs else 2

        with tf.Graph().as_default() as graph:
          dataset  = tf.data.Dataset.from_tensors(preprocess(inputs[0])).repeat()
          iterator = dataset.make_one_shot_iterator()
          return_t = tf.import_graph_def(graph_def=self.graph_def, input_map={self.inputs[0]:
                                                                              iterator.get_next()},
                                         return_elements=outputs, name='')

          # Unwrap the returned output node. For now, we assume we only
          # want the tensor with index `:0`, which is the 0th element of the
          # `.outputs` list.
          output_t = return_t[0].outputs[0]


          ## declare py_functions in graph
          py_func_tokens = _script_ops._py_funcs._funcs.keys()
          for token in py_func_tokens:
            _script_ops._py_funcs.remove(token)
          _script_ops._py_funcs._unique_id = 0
          for pyfunc_name, (partition, input_names, output_dtypes) in self.fpga_pynode_dict.items():
            input_tensors = [graph.get_tensor_by_name(name) for name in input_names]
            fpga_output_tensor = tf.py_func(partition.forward_exec, input_tensors, output_dtypes,
                                            stateful=False, name=pyfunc_name)

          ## declare input and output tensors to network graph
          input_tensors  = [graph.get_operation_by_name(name).outputs[0] for name in self.inputs]
          output_tensors = []
          for output in outputs:
            if isinstance(output, list):
              output = [graph.get_operation_by_name(name).outputs[0] for name in output]
            else:
              output = graph.get_operation_by_name(output).outputs[0]
            output_tensors.append(output)

          import timeit
          with tf.Session(graph=graph, config=config) as sess:
            # create log and inlude graph in the log
            #log_writer = tf.summary.FileWriter('./logs', sess.graph)

            # Warm up run
            print("Warm up run ...")
            for _ in range(10):
              ret = sess.run(output_tensors)

            print("Start timing ...")
            timings = np.zeros(num_iter)
            for i in range(num_iter):
              start_time = timeit.default_timer()
              ret = sess.run(output_tensors)
              end_time = timeit.default_timer()
              timings[i] = end_time - start_time

              print("Iteration {}: {:.6f} s".format(i, end_time - start_time))

            #log_writer.close()

        return ret[0], list(timings)


    def forward_exec(self, inputs, outputs=None, preprocess=None, **kwargs):
        if not outputs:
          outputs = self.outputs
        else:
          for output in outputs:
            if isinstance(output, list):
              raise TypeError('outputs should be flattened list of name strings')

        if not preprocess:
          preprocess = self.preprocess

        config = kwargs['config'] if 'config' in kwargs else None

        if 'sess' in kwargs:
          Graph = kwargs['sess'].graph
        else:
          Graph = tf.Graph()
          with Graph.as_default():
            ## import modified graph
            tf.import_graph_def(self.graph_def, name='')

        with Graph.as_default() as graph:

          ## declare py_functions in graph
          py_func_tokens = _script_ops._py_funcs._funcs.keys()
          for token in py_func_tokens:
            _script_ops._py_funcs.remove(token)
          _script_ops._py_funcs._unique_id = 0
          #with tf.device(global_fpga_device):  ## TODO: add FPGA support for TF
          for pyfunc_name, (partition, input_names, output_dtypes) in self.fpga_pynode_dict.items():
            input_tensors = [graph.get_tensor_by_name(name) for name in input_names]
            fpga_output_tensor = tf.py_func(partition.forward_exec, input_tensors, output_dtypes,
                                            stateful=False, name=pyfunc_name)
          # print('graph py_tunctions:   {}'.format(graph._py_funcs_used_in_graph))
          # print('graph py_func tokens: {}'.format(_script_ops._py_funcs._funcs.keys()))

          ## declare input and output tensors to network graph
          input_tensors  = [graph.get_operation_by_name(name).outputs[0] for name in self.inputs]
          output_tensors = []
          for output in outputs:
            if isinstance(output, list):
              output = [graph.get_operation_by_name(name).outputs[0] for name in output]
            else:
              output = graph.get_operation_by_name(output).outputs[0]
            output_tensors.append(output)

          ## bound the input tensors to input data
          if not isinstance(inputs, list):
            inputs = [inputs]
          feed_dict = {inp_tensor: inp_val for inp_tensor, inp_val in zip(input_tensors, preprocess(inputs))}
          if not feed_dict:
            feed_dict = None

          #with tf.Session(graph=graph, config=tf.ConfigProto(log_device_placement=True)) as sess:
          if 'sess' in kwargs:
            output_values = kwargs['sess'].run(output_tensors, feed_dict=feed_dict)
          else:
            with tf.Session(graph=graph, config=config) as sess:
              output_values = sess.run(output_tensors, feed_dict=feed_dict)

        return {tensor.op.name: val for val, tensor in zip(output_values, output_tensors)}

    def preprocess(self, inputs):
        ## inputs should be a list
        if not isinstance(inputs, list):
          raise TypeError('inputs to preprocessing should be a list')

        res = []
        for inp in inputs:
          if isinstance(inp, np.ndarray):
            res.append(inp)
          elif isinstance(inp, basestring):
            res.append(PyTurboJPEG.imread(inp))
        return res
