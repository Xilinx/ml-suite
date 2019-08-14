#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from os import mkdir as _mkdir
from os.path import exists as _exists

import caffe

from xfdnn.rt.xdnn_rt_base import xdnnRT as _xdnnRT
from xfdnn.rt.xdnn_opt import CPUTransform, HWEmuTransform, FPGATransform
from xfdnn.tools.compile.bin.xfdnn_compiler_caffe import CaffeFrontend





class CaffexdnnRT(_xdnnRT):
    def __init__ (self, args, **kwargs):
        self.inputs = None
        self.outputs = None
        super(CaffexdnnRT, self).__init__(CaffeFrontend, args, **kwargs)

    def load_graph(self, args, **kwargs):
        graph = caffe.Net(args.networkfile, args.weights, caffe.TEST)
        self.save = args.save
        if self.save and not _exists(self.save):
          _mkdir(self.save)

        inputs  = self.inputs if self.inputs else self.list_inputs_of_graph(graph)
        outputs = self.outputs if self.outputs else self.list_outputs_of_graph(graph)
        return graph, inputs, outputs, True

    def list_inputs_of_graph(self, graph):
        res = []
        for name, layer in zip(graph._layer_names, graph.layers) :
          print name, layer.type
          if layer.type in ['Input']:
            res.append(name)
        return res

    def list_outputs_of_graph(self, graph):
        res = []
        bset = set()
        for name, bottoms in graph.bottom_names.items():
            for bottom in bottoms :
                bset.add(bottom)
        for name, tops in graph.top_names.items() :
            for top in tops :
                if top not in bset :
                    res.append(name)
        return res

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
