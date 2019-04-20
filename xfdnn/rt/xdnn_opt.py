#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import copy
import pydot
import sys
from json import loads as _loads
from collections import defaultdict, OrderedDict
from os.path import exists as _exists
from six import string_types as _string_types
import operator, pprint

from os import listdir as _listdir
from os.path import join as _join

import tensor_tools as tt
import keras_tools as kt

from xdnn_env         import xdnn_env as _xdnn_env, xdnn_fpga_env as _xdnn_fpga_env
from factory          import factory as _factory
from conv_layer       import conv_layer as _conv_layer
from eltwise_layer    import eltwise_layer as _eltwise_layer
from scale_layer      import scale_layer as _scale_layer
from concat_layer     import concat_layer as _concat_layer
from identity_layer   import identity_layer as _identity_layer
from pool_layer       import pool_layer as _pool_layer
from reshape_layer    import reshape_layer as _reshape_layer
from matop_layer      import matop_layer as _matop_layer
from quantize_layer   import quantize_layer as _quantize_layer, unquantize_layer as _unquantize_layer
from softmax_layer    import softmax_layer as _softmax_layer
from relu_layer       import relu_layer as _relu_layer
from batchnorm_layer  import batchnorm_layer as _batchnorm_layer
from reduce_layer     import reduce_layer as _reduce_layer
from fpga_pydot_layer import fpga_pydot_layer as _fpga_pydot_layer
from pool_hwemu_layer import pool_hwemu_layer




class available_layers():
  layers = {
    'Convolution':  (_conv_layer, {'mode': 'NCHW'}),
    'BiasAdd':      (_matop_layer, {'optype': 'BiasAdd', 'mode': 'NCHW'}),
    'Eltwise':      (_eltwise_layer, {'operation': 'SUM', 'mode': 'NCHW'}),  # TODO FIXME assumes add???
    #'Mean':         (_reduce_layer, {'type': 'AVG', 'mode': 'NCHW'}),       # TODO FIXME assumes avgpool???
    'Reshape':      (_reshape_layer, {}),
    'Scale':        (_scale_layer, {'mode': 'NCHW'}),
    'ReLU':         (_relu_layer, {}),
    'Pooling':      (_pool_layer, {'mode': 'NCHW'}),
    'Concat':       (_concat_layer, {'axis': 1}),
    'BatchNorm':    (_batchnorm_layer, {}),
    'InnerProduct': (_matop_layer, {'optype': 'MatMul'}),
    # 'Mul':         (_matop_layer, {'optype': 'MatMul'}),
    'Sub':          (_matop_layer, {'optype': 'Sub'}),
    'Identity':     (_identity_layer, {}),
    'Dropout' :     (_identity_layer, {}),
    'Input' :       (_identity_layer, {}),
    'Output' :      (_identity_layer, {}),
    'Softmax':      (_softmax_layer, {})
  }

  def _ret(self, name, kwargs={}):
    layer, defaults = self.layers[name]
    ## update available_layers arguments based on kwargs
    defaults.update(kwargs)
    return layer(**defaults)

  def __call__(self, name, **kwargs):
    return self._ret(name, kwargs)

  def __getitem__(self, name):
    return self._ret(name)

  def __contains__(self, other):
    if isinstance(other, _string_types):
      return other in self.layers
    else:
      return False



_available_layers = available_layers()





class CPUTransform:

  def __init__(self, time_to_layer_list=None, layerparameter_dict=None, options=object(), native_graph=None, networkjson = None, weightdir = None, inps = None, outs = None):
    self.variables = {}
    self.constSet = set()
    if networkjson == None :
      self.orig_framework = options.base
      self.native_graph = native_graph
      self._layers = self.create_schedule(time_to_layer_list, layerparameter_dict, options)
    else :
      self._layers = self.create_compiled_schedule(networkjson, weightdir, inps)

  def extract_sub_graph(self, nodes, layerparameter_dict):
    sub_graph = set()
    stk = list(nodes)
    while len(stk) > 0:
      node_name = stk.pop()
      sub_graph.add(node_name)
      if node_name in layerparameter_dict:
        params = layerparameter_dict[node_name]
        if params.bottoms is not None:
          stk += [inp for inp in params.bottoms if inp not in sub_graph]
    return sub_graph

  def create_compiled_schedule(self, networkjson, weightdir, inps) :
    import numpy as np
    weights = _listdir(weightdir)
    weights = [_join(weightdir, wt) for wt in weights]
    const = {}
    for wtfile in weights :
      with open(wtfile, 'r') as wts :
        line = wts.readline()
        toks = line.strip().split()
        print len(toks)
        if len(toks) > 4 :
          print toks[0], toks[1], toks[2], toks[3], len(toks[4:])
          if toks[0] not in const :
            const[toks[0]] = {}
          if "bias" in wtfile[wtfile.rfind('/'):] :
            const[toks[0]]['bias'] = np.array([float(x) for x in toks[4:]])
          else :
            const[toks[0]]['weights'] = np.array([float(x) for x in toks[4:]])
    schedule = []
    for layer in networkjson['network'] :
      if layer['type'] in _available_layers :
        print layer['type'], layer['name'], layer['bottoms'], layer['type'] not in ['Convolution', 'InnerProduct'] or (layer['name'] in const and len(const[layer['name']]) == 2)
        xdlf_layer = copy.deepcopy(_available_layers(layer['type'], mode='NCHW'))
        if layer['name'] in const :
          xdlf_layer.set_layer_params(layer, const[layer['name']])
        else :
          xdlf_layer.set_layer_params(layer)
        schedule.append(xdlf_layer)
      elif layer['name'] in inps :
        print "Detected input : ", layer['name'], layer['type'], layer['outputshapes']
    print schedule     
    return schedule   

  def create_schedule(self, time_to_layer_list, layerparameter_dict, options):
    print("Creating schedule for \"CPU\"")
    #print('processing layers: {}'.format(zip(*time_to_layer_list)[1]))

    schedule = []
    print('time, layer_type,      layer_name,                 layer_inputs')
    for t, layer_name in time_to_layer_list:
      layer_params = layerparameter_dict[layer_name]

      print('{:3d}, {:15s}, {:s},     {}'.format(t, layer_params.type[0], layer_params.name,
                      layer_params.bottoms))

      if layer_params.type[0] == 'Const':
        self.variables[layer_params.name] = layer_params.data
      elif layer_params.type[0] in _available_layers:
        ## here mode is PFGA IP data_format (not the platform data_format, i.e., options.data_format)
        layer = copy.deepcopy(_available_layers(layer_params.type[0], mode='NCHW'))
        layer.set_params(layer_params, self.variables)
        layer.name = layer_name
        schedule.append(layer)
      # elif layer_params.type[0] in ['Input', 'Output']:
      #   ## NOTE: This is to ignore unrecognized Input and Output nodes created by compiler with
      #   ##       --cpulayermustgo flag
      #   pass
      else:
        layer = self.get_default_layer(layer_params)
        layer.name = layer_name
        schedule.append(layer)
    self.constSet = set(self.variables.keys())

    return schedule

  def get_default_layer(self, layer_params):
    if self.orig_framework.lower() == 'tf' :
        ## FIXME: Hack to by pass caffe and tensorflow co-existance issues
        from layer_tf import layer_tf as _layer_tf
        l = _layer_tf(layer_params.bottoms, layer_params.tops[0], self.native_graph, 'NCHW')
        l.get_constant_inputs(self.constSet)
    elif self.orig_framework.lower() == "caffe" :
        ## FIXME: Hack to by pass caffe and tensorflow co-existance issues
        from layer_caffe import layer_caffe as _layer_caffe
        l = _layer_caffe(layer_params.bottoms, layer_params.tops[0], self.native_graph, 'NCHW')
        l.get_constant_inputs(self.constSet)
    else :
        print ("framework not yet supported")
    return l

  def getLayers(self):
    return self._layers

  def getLayerNames(self):
    return [layer.output for layer in self._layers]

class FPGATransform (CPUTransform):
  def __init__(self, time_to_layer_list, layerparameter_dict, compilerJson, options=object(),
               native_graph=None, name_postfix=None):
    CPUTransform.__init__(self, time_to_layer_list, layerparameter_dict, options, native_graph)

    print("Creating schedule for \"FPGA\"")

    self.name_postfix = name_postfix
    self.fpga_layer_cnt = 0

    if isinstance(options.fpga_recipe, dict):
      recipe = options.fpga_recipe
    elif isinstance(options.fpga_recipe, _string_types):
      recipe = _loads(options.fpga_recipe)
    else:
      recipe = {"start": [], "end": []}

    self._boundryMap = {'inputs':  compilerJson.get('inputs', []),
                        'outputs': compilerJson.get('outputs', [])}

    ## NOTE: each layer might have multiple commands due to gather and scatter
    self._layerParameterMap = defaultdict(list)
    for l in compilerJson['network']:
      self._layerParameterMap[l['name']] += [l]

    self.xdnn_env = _xdnn_fpga_env(options.xclbin, quant_cfgfile=options.quant_cfgfile, isxdnnv3=(options.xdnnv3==True))

    newSchedule = []  # we will be building this
    layersForFpga = []
    collectLayersForFpga = False
    for li, ol in enumerate(self._layers):
      if ((not recipe["start"] and isinstance(ol, _conv_layer))
          or ol.output in recipe["start"]):
        collectLayersForFpga = True

      if ol.output in self._layerParameterMap and collectLayersForFpga:
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

    # update schedule with new FPGA schedule
    self._layers = newSchedule

  def _make_fpga_layer(self, layersForFpga):
    compilerInfo = OrderedDict()
    for ol in layersForFpga:
      ol_name = ol.output
      compilerInfo[ol_name] = {}
      compilerInfo[ol_name]['layerParameter'] = self._layerParameterMap[ol_name]
      compilerInfo[ol_name]['weights'] = ol.filter_weights if hasattr(ol, "filter_weights") else None
      compilerInfo[ol_name]['biases'] = ol.biases if hasattr(ol, "biases") else None

    l = _fpga_pydot_layer(compilerInfo=compilerInfo, boundryMap=self._boundryMap,
                          xdnn_env=self.xdnn_env, name_postfix=self.name_postfix)

    l.name = 'fpga_pydot_layer_{}'.format(self.fpga_layer_cnt)
    self.fpga_layer_cnt += 1

    l.setInput([input['input_name'] for input in self._boundryMap['inputs']])
    l.setOutput([output['previous_layers'][0] for output in self._boundryMap['outputs']])
    return l

class HWEmuTransform(CPUTransform):
  def __init__(self, time_to_layer=None, layer_param_dict=None, options=object(), native_graph=None, networkjson = None, weightdir = None, inps = None, outs = None, isV3 = True):
    print("Creating schedule for \"HWEmu\"")
    CPUTransform.__init__(self, time_to_layer, layer_param_dict, options, native_graph, networkjson, weightdir, inps, outs)
    if networkjson :
      recipe = {}
      recipe['start'] = inps
      recipe['end'] = outs
      recipe['quantize'] = {inp:inp for inp in inps}
      recipe['unquantize'] = {out:out for out in outs}
      doQuantizeSubgraph = True
      xdnnv3 = isV3
      self.xdnn_env = _xdnn_env()

    else :
      recipeStr = options.quant_recipe
      print recipeStr
      recipe = json.loads(recipeStr)
      doQuantizeSubgraph = False # flag that we need to quantize the subgraph
      self.xdnn_env = None
      if options.quant_cfgfile[-4:].lower() == 'json' :
        self.xdnn_env = _xdnn_env(options.quant_cfgfile)
      else :
        self.xdnn_env = _xdnn_env()
      xdnnv3 = options.xdnnv3

    if xdnnv3 == True:
      opFactorySelect = "hwEmuV3"
    else:
      opFactorySelect = "hwEmuV2"
    
    self._is_var_quantized = set()
    quantizedLayers = defaultdict(int) # for stats
    newSchedule = []  # we will be building this

    print recipe
    print recipe['quantize']
    print recipe['unquantize']
    for i, ol in enumerate(self._layers):
      origLayerName = ol.output

      if origLayerName == recipe['start']:
        # 'start' signal to quantize subgraph until 'end' signal
        doQuantizeSubgraph = True

      if origLayerName in recipe['quantize'] and ol.deephi_quantizations is None :
        # inject quantize_layer
        print("Start quantize subgraph @ {:s}".format(origLayerName))
        quantizeKey = recipe['quantize'][origLayerName]
        l = _quantize_layer(quantizeKey, self.xdnn_env)
        l.setInput([ol.inputs[0]])
        l.setOutput(self._quantized_varname(ol.inputs[0]))

        newSchedule.append(l)

      if doQuantizeSubgraph:
        # substitute layers in quantized subgraph
        if isinstance(ol, _conv_layer):
          l = _factory.conv_factory(
              opFactorySelect,
              weights=ol.filter_weights,
              stride=ol.conv_stride,
              activation=ol.activation_fn,
              padding_type=ol.padding_type,
              paddings=ol.paddings,
              biases=ol.biases,
              mode=ol.mode,
              quantize_key=ol.deephi_quantizations if ol.deephi_quantizations else origLayerName,
              #quantize_key= origLayerName,
              isV3=xdnnv3,
              xdnn_env=self.xdnn_env)
        elif isinstance(ol, _eltwise_layer):
          l = _factory.eltwise_factory(
              opFactorySelect,
              operation=ol.operation,
              activation=ol.activation,
              mode=ol.mode,
              quantize_key=ol.deephi_quantizations if ol.deephi_quantizations else origLayerName,
              isV3=xdnnv3,
              xdnn_env=self.xdnn_env)
        elif isinstance(ol, _scale_layer):
          l = _factory.scale_factory(
              opFactorySelect,
              alpha=ol.alpha,
              beta=ol.beta,
              activation=ol.activation,
              mode=ol.mode,
              quantize_key=ol.deephi_quantizations if ol.deephi_quantizations else origLayerName,
              isV3=xdnnv3,
              xdnn_env=self.xdnn_env)
        elif isinstance(ol, _relu_layer):
          l = ol
        elif isinstance(ol, _pool_layer):
          l = _factory.pool_factory(
                opFactorySelect,
                pool_type=ol.pool_type,
                dim=ol.pool_kernel_dim,
                stride=ol.pool_stride,
                padding_type=ol.padding_type,
                paddings=ol.paddings,
                mode=ol.mode,
                global_pooling=ol.global_pooling,
                quantize_key=ol.deephi_quantizations if ol.deephi_quantizations else origLayerName,
                xdnn_env=self.xdnn_env)
        elif (isinstance(ol, _matop_layer)
              and ol.optype == "Add"):
          l = ol
        elif (isinstance(ol, _concat_layer)
              or (hasattr(ol, "op") and "Concat" in ol.op.type)):
          l = ol
        else:
          raise NotImplementedError('unknown layer quantizer {:s} {:s}'.format((origLayerName,type(ol))))

        quantizedLayers[type(ol)] += 1

        # reroute to used quantized vars, then add to new schedule
        l.setInput(list(map(self._quantized_varname, ol.inputs)))
        l.setOutput(self._quantized_varname(origLayerName))
        if ol.deephi_quantizations :
          if doQuantizeSubgraph :
            l.deephi_quantizations = ol.deephi_quantizations
          else :
            l.deephi_quantizations = None
        newSchedule.append(l)
      else:
        # add new schedule as-is
        newSchedule.append(ol)

      if origLayerName in recipe['unquantize'] and ol.deephi_quantizations is None :
        # inject unquantize_layer
        print("End quantize subgraph @ {:s}".format(origLayerName))
        quantizeKey = recipe['unquantize'][origLayerName]
        l = _unquantize_layer(quantizeKey, self.xdnn_env)
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
    print newSchedule
    self._layers = newSchedule

  def _quantized_varname(self, x):
    if x in self._is_var_quantized:
      return x + '_quantized'
    return x
