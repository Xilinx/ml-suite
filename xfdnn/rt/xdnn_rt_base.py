#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import json
from copy import deepcopy
from collections import defaultdict
from six import string_types as _string_types

import numpy as np

from xfdnn.rt import xdnn_util





class LayerPartition(object):
  def __init__(self, index=0, supported=False, time_to_layer_list=None, layerparameter_dict={},
               layeroutput_dict={}, spt_set={}):
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

    self.spt_set = {extra: layer for extra, layer in spt_set.items() if layer in self.names}

    self.inputs,    \
    self.outputs,   \
    self.inputMap,  \
    self.outputMap, \
    self.input_cnt = self.partition_boundries(layerparameter_dict, layeroutput_dict)

  def size(self):
    return len(self.names)

  def partition_boundries(self, layerparameter_dict, layeroutput_dict):
    '''
    returns sourcenodes, sinknodes.
    sourcenodes are exclusive, i.e., they do not belong to the layer, except when the sourcenode
    does not have an input itself.
    sinknodes are inclusive, i.e., they are part of the layer.
    sourcenode_map are the last node that was collapsed into the sourcenodes.
    sinknode_map are the last node that was collapsed into the sinknodes.
    '''
    partition_name_set = set(self.names)
    sourcenodes = []
    sinknodes   = []
    bottoms     = []
    for layer in self.names:
      inps = layerparameter_dict[layer].bottoms
      if inps is None or len(inps) == 0:
        sourcenodes += [layer]
      else:
        bottoms += inps

      outs = layeroutput_dict[layer].keys()
      ## NOTE: "not outs" might cause issues
      if not outs or any([output not in partition_name_set for output in outs]):
        sinknodes += [layer]

    sourcenodes += [bottom for bottom in bottoms if bottom not in partition_name_set]
    # sinknodes    = [layer for layer in self.names if layer not in set(bottoms)]   ## deprecated

    sourcenodes_cnt = defaultdict(lambda: 0)
    for sourcenode in sourcenodes:
      sourcenodes_cnt[sourcenode] += 1

    sourcenodes = sourcenodes_cnt.keys()
    # sinknodes   = list(set(sinknodes))

    sourcenode_map = {}
    for sourcenode in sourcenodes:
      collapse_future = layerparameter_dict[sourcenode].collapse_future
      if not collapse_future or sourcenode in partition_name_set:
        sourcenode_map[sourcenode] = sourcenode
      else:
        #collapse_future = [extra if isinstance(extra, _string_types) else extra.name for extra in collapse_future]
        sourcenode_map[sourcenode] = collapse_future[-1]

    sinknode_map = {}
    for sinknode in sinknodes:
      collapse_future = layerparameter_dict[sinknode].collapse_future
      if not collapse_future:
        sinknode_map[sinknode] = sinknode
      else:
        #collapse_future = [extra if isinstance(extra, _string_types) else extra.name for extra in collapse_future]
        sinknode_map[sinknode] = collapse_future[-1]
    return sourcenodes, sinknodes, sourcenode_map, sinknode_map, sourcenodes_cnt

  def update(self, layerparameter_dict, layeroutput_dict, others):
    if not isinstance(others, list):
      others = [others]
    for other in others:
      other.fValid   = False
      self.schedule += other.schedule
      self.names    += other.names
      self.spt_set.update(other.spt_set)
    self.inputs,    \
    self.outputs,   \
    self.inputMap,  \
    self.outputMap, \
    self.input_cnt = self.partition_boundries(layerparameter_dict, layeroutput_dict)

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

    for layer in self.layers:
      #print('Forward Exec: {}'.format(layer.name))
      layer_inputs = [self.variables[inp] for inp in layer.inputs]
      if layer_inputs:
        layer_outputs = layer.forward_exec( layer_inputs )

        ## FIXME: HACK until all cpu layer outputs have been updated to dictionary
        if not isinstance(layer_outputs, dict):
          if isinstance(layer.output, list) ^ isinstance(layer_outputs, list):
            raise RuntimeError('Expected output of type {} received {}'.format(type(layer.output), type(layer_outputs)))
          elif isinstance(layer.output, list) and isinstance(layer_outputs, list):
            layer_outputs = {name: output for name, output in zip(layer.output, layer_outputs)}
          else:
            layer_outputs = {layer.output: layer_outputs}

        self.variables.update(layer_outputs)
        # for name in layer_outputs.keys():
        #   print('Forward Exec: {} Dimensions: {}'.format(name, layer_outputs[name].shape))
      # elif len(inputs) == 1:
      #   self.variables[layer.output] = inputs[0]
      else:
        raise RuntimeError('layer input mismatch')

    if save is not None:
      for var in self.variables.keys():
        #np.save(save+'/'+'_'.join(var.split('/'))+'.npy', self.variables[var])
        np.savetxt(save+'/'+'_'.join(var.split('/'))+'.txt', self.variables[var].flatten(),fmt="%0.6f")

    return [self.variables[name] for name in outputs]








class xdnnRT(object):
    def __init__(self, compilerFunc, args, **kwargs):
        args = xdnn_util.dict2attr(args)
        args.update(kwargs)

        self._args = args

        if args.networkfile is None and args.loadpickle is None:
          raise AttributeError('network argument is missing')

        self._graph, \
        self.inputs, \
        self.outputs, \
        self.fValidGraph \
            = self.load_graph(args,
                              startnode=None,
                              finalnode=None,
                              inclusive=True,
                              fixinputnames=False,
                              remove_training_nodes=True,
                              remove_redundant_nodes=True
                             )

        self.fPartition    = args.get('fPartition', True)
        self.networkfile   = args.networkfile
        self.data_format   = args.data_format
        self.save2modeldir = args.save2modeldir   # whether to save to mo directory

        name_postfix       = args.get('name_postfix', '_frozen')
        self.picklefile    = self.file_path('.pickle', name_postfix=name_postfix)   # save pydotGraph and compilerJson

        ## run compiler
        compiler = compilerFunc(args,
                                fixinputnames=True,
                                savepickle=self.picklefile if args.savepickle else None,
                                loadpickle=self.picklefile if args.loadpickle else None,
                               )

        self._compiler_graph, \
        self._compiler_inputs, \
        self._compiler_outputs, \
        pydotGraph, \
        compilerSchedule, \
        ssize, \
        compilerJson \
            = compiler.compile()

        if compilerJson is None:
          raise RuntimeError('Compiler failed to produce valid schedule')

        self.pydotGraph = pydotGraph
        self.compilerSchedule = compilerSchedule
        self.compilerJson = compilerJson

        ## store compiler json
        with open(self.file_path('.json', name_postfix=name_postfix+'-xdlfCompiler'), "w") as f:
          json.dump(compilerJson, f, sort_keys=True, indent=4, separators=(',',': '))


        self.spt_set, \
        self.unspt_set, \
        self.layerparameter_dict, \
        self.layeroutput_dict, \
        self.layer_spt_dict = self.analyze_compiler_output(pydotGraph.get_nodes(),
                                                           compilerJson['unsupported'],
                                                           compilerSchedule.time_to_layer)

        if not self.fPartition:
          return

        ## partitioning the graph
        self.graph_partitions = []
        ## Discover partitions based on compiler schedule
        self.graph_partitions = self.partitions_from_compiler_schedule()

        ## optimize partitions
        self.graph_partitions = self.refine_graph_partitions(self.graph_partitions)

        # self.fpga_partition_cnt = 0
        # for partition in self.graph_partitions:
        #   if partition.supported:
        #     self.fpga_partition_cnt += 1

        ## device transformation for supported layers
        self.device_transforms(args)

        self.rebuild_graph()


    def file_path(self, file_extension, name_prefix='', name_postfix=''):
        path_split = self.networkfile.rsplit('/', 1)
        savepath = path_split[0] if self.save2modeldir and len(path_split) > 1 else '.'
        networkname = path_split[-1].rsplit('.', 1)[0]
        return '/'.join([savepath, name_prefix + networkname + name_postfix + file_extension])

    def load_graph(self, args, **kwargs):
        raise NotImplementedError('load_graph method not implemented for xdnnRT')

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

    def analyze_compiler_output(self, pydotGraph_nodes, compilerJson_unsupported, time_to_layer):
        spt_set             = {}
        unspt_set           = set()
        layerparameter_dict = {}
        layeroutput_dict    = defaultdict(lambda: {})
        layer_spt_dict      = defaultdict(lambda: [0, []])    # NOTE: format: {grp_idx: (spt_flag, t2l_list)}

        for node in pydotGraph_nodes:
          LayerParameter = node.get('LayerParameter')
          layer_name = LayerParameter.name
          layerparameter_dict[layer_name] = LayerParameter
          if 'blob' not in layer_name and LayerParameter.bottoms:       ## FIXME: hack to remove output blobs from output list
            for input_index, input_name in enumerate(LayerParameter.bottoms):
              layeroutput_dict[input_name][layer_name] = input_index

        unspt_set = set(compilerJson_unsupported['list'].keys()) # build unsupported layer set
        ## partition consecutive supported layers in zip(*list(partition.next()[1]))[1]
        print('.... is_supported, layer_index, layer_name')
        partition_idx   = -1
        prev_spt        = True if time_to_layer[0][0] in unspt_set else False
        for time, layers in time_to_layer.items():
          for layer in layers:
            spt = False if layer in unspt_set else True
            if spt ^ prev_spt:
              partition_idx += 1
              layer_spt_dict[partition_idx][0] = spt
            layer_spt_dict[partition_idx][1] += [(time, layer)]
            prev_spt = spt

            ## build supported layer set
            if spt:
              spt_set[layer] = layer
              extras_and_future = layerparameter_dict[layer].extras_and_future
              if extras_and_future is not None:
                extras_names = [extra if isinstance(extra, _string_types) else extra.name for extra in extras_and_future]
                spt_set.update({name: layer for name in extras_names if name not in unspt_set})

            print('.... {0:5s}, {1:3d}, {2:s}'.format(str(spt), time, layer))

        return spt_set, unspt_set, layerparameter_dict, layeroutput_dict, layer_spt_dict

    def partitions_from_compiler_schedule(self):
        print('\nPartition FPGA (un)supported layers from compiler schedule ....')

        layerparameter_dict = self.layerparameter_dict
        layeroutput_dict    = self.layeroutput_dict
        spt_set             = self.spt_set
        layer_spt_dict      = self.layer_spt_dict

        graph_partitions = [LayerPartition(grp_idx, spt, t2l_list, layerparameter_dict,
                                           layeroutput_dict, spt_set) for grp_idx, (spt, t2l_list)
                            in layer_spt_dict.items()]

        print('Partition FPGA (un)supported layers from compiler schedule [DONE]')
        return graph_partitions


    def refine_graph_partitions(self, graph_partitions):
        def partitions_connectivity(graph_partitions, debugprint=False):
            # print('partition connectivity:')
            connectivity         = defaultdict(lambda: defaultdict(list))
            reverse_connectivity = defaultdict(lambda: defaultdict(list))
            for index, partition in enumerate(graph_partitions):
              other_partitions = list(graph_partitions)
              other_partitions.pop(index)
              for other in other_partitions:
                temp_set = set(other.inputs).intersection(set(partition.names))
                if temp_set:
                  for name in temp_set:
                    connectivity[partition.index][name] += [other.index for i in
                                                           range(other.input_cnt[name])]

                temp_set = set(partition.inputs).intersection(set(other.names))
                if temp_set:
                  for name in temp_set:
                    reverse_connectivity[partition.index][name] += [other.index for i in
                                                                    range(partition.input_cnt[name])]

            if debugprint:
              for index, partition in enumerate(graph_partitions):
                print('.... partition ({:3d}, {:5s}) --> {}'.format(partition.index, str(partition.supported),
                                                                    [(graph_partitions[index].index,
                                                                      graph_partitions[index].supported) for
                                                                     index_list in
                                                                     connectivity[partition.index].values()
                                                                     for index in index_list]))
              print('....')
              for index, partition in enumerate(graph_partitions):
                print('.... partition ({:3d}, {:5s}) <-- {}'.format(partition.index, str(partition.supported),
                                                                    [(graph_partitions[index].index,
                                                                      graph_partitions[index].supported) for
                                                                     index_list in
                                                                     reverse_connectivity[partition.index].values()
                                                                     for index in index_list]))
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
              for __, v_list in connectivity_map[vertex].items():
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
              for __, v_list in connectivity[vertex].items():
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
                    end_idx   = ordering.index(src)
                    start_idx = ordering.index(dst)
                    return set(ordering[start_idx:end_idx+1])

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

            #############################
            ## acyclic_matching body
            #############################
            connectivity, \
            reverse_connectivity = partitions_connectivity(graph_partitions)

            topOrder = topological_ordering(connectivity, reverse_connectivity)

            matching = xdnn_util.UnionFind(len(topOrder))
            for src_v in topOrder[::-1]:
              src_support = graph_partitions[src_v].supported
              for __, v_list in connectivity[src_v].items():
                for dst_v in v_list:
                  dst_support = graph_partitions[dst_v].supported
                  if src_support == dst_support:
                    if is_contraction_valid(connectivity, reverse_connectivity, topOrder, src_v, dst_v):
                      matching.union(src_v, dst_v)
            merge_matches = matching.components()

            graph_partitions = merge_partitions(graph_partitions, list(merge_matches.items()))
            return graph_partitions


        #############################
        ## refine_graph_partitions body
        #############################
        print('\nRefine Graph Partitions ....')

        graph_partitions = acyclic_matching(graph_partitions)

        self.connectivity, \
        self.reverse_connectivity = partitions_connectivity(graph_partitions, debugprint=True)

        #for partition in graph_partitions:
          ## FIXME: Heuristic. Don't use fpga for small partitions
          #if partition.size() < 4:
          #  partition.supported = False

        print('\nSUMMARY:')
        for partition in graph_partitions:
          print('.... partition_index \"{}\"'.format(partition.index))
          print('.... inputs:          {}'.format(partition.inputMap.keys()))
          print('.... inputs actual:   {}'.format(partition.inputMap.values()))
          print('.... outputs:         {}'.format(partition.outputMap.keys()))
          print('.... outputs actual:  {}'.format(partition.outputMap.values()))
        print('Refine Graph Partitions [DONE]')

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
        while ctr < len(img_list):
            ctrmax = min(ctr+batch, len(img_list))
            print("processing", img_list[ctr:ctrmax])
            res = self.forward_exec(img_list[ctr:ctrmax], preprocess=preprocess)
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
