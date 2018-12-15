#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import copy
import pydot
import json
import sys
from collections import defaultdict
import operator, pprint

import tensor_tools as tt
import keras_tools as kt

import layer
import xdnn_env
from factory import factory
from conv_layer import conv_layer
from eltwise_layer import eltwise_layer
from scale_layer import scale_layer
from concat_layer import concat_layer
from identity_layer import identity_layer
from pool_layer import pool_layer
from reshape_layer import reshape_layer
from matop_layer import matop_layer
from quantize_layer import quantize_layer, unquantize_layer
from softmax_layer import softmax_layer
from relu_layer import relu_layer
from batchnorm_layer import batchnorm_layer
from layer_tf import layer_tf
from reduce_layer import reduce_layer
from fpga_pydot_layer import fpga_pydot_layer


class CPUTransform:

  def __init__(self, inputs, graph, schedule, options=object()):
    self.variables = {}
    self.constSet = set()

    self.available_layers = {
      'Convolution': conv_layer(mode='NCHW'),  # done
      'BiasAdd': matop_layer('BiasAdd', mode="NCHW"),  # done
      'Eltwise': eltwise_layer(operation = 'SUM', mode = 'NCHW'),  # TODO FIXME assumes add???
      # 'Mean': reduce_layer('AVG', mode='NCHW'), # TODO FIXME assumes avgpool???
      'Reshape': reshape_layer(),
      'Scale': scale_layer(mode = 'NCHW'),  # done
      'ReLU': relu_layer(),  # done
      'Pooling': pool_layer(mode='NCHW'),  # done
      'Concat': concat_layer(1),  # done
      'BatchNorm': batchnorm_layer(),  # done
      'InnerProduct': matop_layer('MatMul'),  # done
      # 'Mul': matop_layer('MatMul'), #done
      'Sub': matop_layer('Sub'),  # done
      'Identity':  identity_layer(),
      'Softmax': softmax_layer()
    }

    ignore = self.compute_ignore_nodes(inputs, graph)
    self._layers = self.create_schedule(graph, ignore, schedule)

  def compute_ignore_nodes(self, nodes, graph):
    ignore = set()
    stk = nodes[:]
    while len(stk) > 0:
      node_name = stk.pop()
      ignore.add(node_name)
      g_node = graph.get_node(pydot.quote_if_necessary(node_name))
      if len(g_node) > 0:
        g_node = g_node[0]
        params = g_node.get('LayerParameter')
        if params.bottoms is not None:
          stk += [inp for inp in params.bottoms if inp not in ignore]
    return ignore

  def create_schedule(self, graph, ignore, sch):
    print("CBDG: Creating schedule")
    schedule = []
    objmap = {}
    print('ignores:{}'.format(ignore))
    for node in graph.get_nodes():
      node_name = node.get('LayerParameter').name
      objmap[node_name] = node

    print('time, layer_type, layer_name, layer_inputs')
    for t, layer_name in sch.time_to_layer.items():
      # print(t, layer_name)
      # assuming that there is only one operation happening per time.
      layer_params = objmap[layer_name[0]].get('LayerParameter')

      #print "\nANDBG create_schedule %s %s" % (layer_params.name, layer_params.type)
      #print layer_params # ANDBG
      #print layer_params.layer[0]

      print('{:3d}, {:15s}, {:s}, {}'.format(t, layer_params.type[0], layer_params.name,
                      layer_params.bottoms))
      if layer_params.type[0] == 'Const':
        self.variables[layer_params.name] = layer_params.data
      elif layer_params.name not in ignore:
        if layer_params.type[0] in self.available_layers:
          layer = copy.deepcopy(self.available_layers[layer_params.type[0]])
          layer.set_params(layer_params, self.variables)
          schedule.append(layer)
        else:
          schedule.append(self.get_default_layer(layer_params))
    self.constSet = set(self.variables.keys())

    return schedule

  def get_default_layer(self, layer_params):
    l = layer_tf(layer_params.bottoms, layer_params.tops[0], layer_params.layer[0].graph, 'NCHW')
    l.get_costant_inputs(self.constSet)
    return l

  def getLayers(self):
    return self._layers

class FPGATransform (CPUTransform):
  def __init__(self, inputs, graph, schedule, compilerJson, options=object()):
    xclbin = options.xclbin
    recipe = {"start": [], "end": []}
    if options.fpga_recipe:
      recipe = json.loads(options.fpga_recipe)
    self._compilerJson = compilerJson
    self._compilerLayerMap = {}
    for l in self._compilerJson["network"]:
      self._compilerLayerMap[l["name"]] = l
    with open("xdlf_compiler.json", "w") as f:
      json.dump(self._compilerJson, f, 
        sort_keys=True, indent=4, separators=(',',': '))

    CPUTransform.__init__(self,inputs, graph, schedule, options)
    self.xdnnEnv = xdnn_env.xdnn_fpga_env(xclbin, options.xdnnv3 == True)

    newSchedule = []  # we will be building this 
    layersForFpga = []
    collectLayersForFpga = False
    for li, ol in enumerate(self._layers):
      if (not recipe["start"] and isinstance(ol, conv_layer)) \
        or ol.output in recipe["start"]:
        collectLayersForFpga = True

      if ol.output in self._compilerLayerMap and collectLayersForFpga:
        layersForFpga.append(ol)
      else:
        newSchedule.append(ol)

      if not recipe["end"] or ol.output in recipe["end"]:
        if layersForFpga:
          newSchedule.append(self._make_fpga_layer(layersForFpga))
          layersForFpga = []
        collectLayersForFpga = False

    if layersForFpga:
      newSchedule.append(self._make_fpga_layer(layersForFpga))
      layersForFpga = []

    # update schedule with new FPGA schedule
    self._layers = newSchedule

  def _make_fpga_layer(self, layersForFpga):
    weights = []
    biases = []
    compilerLayers = []
    for ol in layersForFpga:
      if hasattr(ol, "filter_weights"):
        weights.append(ol.filter_weights)
      else:
        weights.append(None)
      if hasattr(ol, "biases"):
        biases.append(ol.biases)
      else:
        biases.append(None)
      compilerLayers.append(self._compilerLayerMap[ol.output])

    l = fpga_pydot_layer(\
        weights=weights,
        biases=biases,
        compilerLayers=compilerLayers,
        xdnn_env=self.xdnnEnv)
    l.setInput(layersForFpga[0].inputs)
    l.setOutput(layersForFpga[-1].output)
    return l
    
class HWEmuTransform(CPUTransform):
  def __init__(self, inputs, graph, schedule, options=object()):
    recipeStr = options.quant_recipe
    recipe = json.loads(recipeStr)

    if options.xdnnv3 == True:
      opFactorySelect = "hwEmuV3"
    else:
      opFactorySelect = "hwEmuV2"

    CPUTransform.__init__(self, inputs, graph, schedule, options)
    self.xdnnEnv = xdnn_env.xdnn_env()
    self._is_var_quantized = set()

    quantizedLayers = defaultdict(int) # for stats
    doQuantizeSubgraph = False # flag that we need to quantize the subgraph
    newSchedule = []  # we will be building this
    for i, ol in enumerate(self._layers):
      origLayerName = ol.output

      if origLayerName == recipe['start']:
        # 'start' signal to quantize subgraph until 'end' signal
        doQuantizeSubgraph = True

      if origLayerName in recipe['quantize']:
        # inject quantize_layer
        print("Start quantize subgraph @ {:s}".format(origLayerName))
        quantizeKey = recipe['quantize'][origLayerName]
        l = quantize_layer(quantizeKey, self.xdnnEnv)
        l.setInput([ol.inputs[0]])
        l.setOutput(self._quantized_varname(ol.inputs[0]))

        newSchedule.append(l)

      if doQuantizeSubgraph:
        # substitute layers in quantized subgraph
        if isinstance(ol, conv_layer):
          l = factory.conv_factory(
            opFactorySelect,
            ol.filter_weights,
            ol.conv_stride,
            ol.activation_fn,
            ol.padding,
            ol.biases,
            ol.mode,
            origLayerName,
            self.xdnnEnv)
        elif isinstance(ol, eltwise_layer):
          l = factory.eltwise_factory(
            opFactorySelect,
            ol.operation,
            ol.activation,
            ol.mode,
            origLayerName,
            self.xdnnEnv)
        elif isinstance(ol, scale_layer):
          l = factory.scale_factory(
            opFactorySelect,
            ol.activation,
            ol.mode,
            origLayerName,
            self.xdnnEnv)
        elif isinstance(ol, relu_layer):
          l = ol
        elif isinstance(ol, pool_layer):
          l = ol
        elif (isinstance(ol, matop_layer)
              and ol.optype == "Add"):
          l = ol
        elif (isinstance(ol, concat_layer)
              or (hasattr(ol, "op") and "Concat" in ol.op.type)):
          l = ol
        else:
          raise NotImplementedError('unknown layer quantizer {:s} {:s}'.format((origLayerName,type(ol))))

        quantizedLayers[type(ol)] += 1

        # reroute to used quantized vars, then add to new schedule
        l.setInput(list(map(self._quantized_varname, ol.inputs)))
        l.setOutput(self._quantized_varname(origLayerName))
        newSchedule.append(l)
      else:
        # add new schedule as-is
        newSchedule.append(ol)

      if origLayerName in recipe['unquantize']:
        # inject unquantize_layer
        print("End quantize subgraph @ {:s}".format(origLayerName))
        quantizeKey = recipe['unquantize'][origLayerName]
        l = unquantize_layer(quantizeKey, self.xdnnEnv)
        l.setInput([self._quantized_varname(origLayerName)])
        l.setOutput(origLayerName)
        newSchedule.append(l)

      if origLayerName == recipe['end']:
        # 'end' signal to stop quantizing subgraph
        doQuantizeSubgraph = False

    print("Quantized layers:")
    sortedQL = sorted(list(quantizedLayers.items()), key=operator.itemgetter(1))
    print(pprint.pprint(sortedQL))

    # update schedule with new quantized schedule
    self._layers = newSchedule

  def _quantized_varname(self, x):
    if x in self._is_var_quantized:
      return x + '_quantized'
    return x
