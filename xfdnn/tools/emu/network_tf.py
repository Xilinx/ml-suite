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
import numpy as np
import copy

import layer
import conv_layer
import concat_layer
import identity_layer
import pool_layer
import reshape_layer
import matop_layer
import softmax_layer
import relu_layer
import batchnorm_layer

import layer_tf

import xdnn_env
import conv_hwemu_layer
import matop_hwemu_layer
import bunch_hwemu_layer

import quantize_layer
import quantize_instrument_layer
import operator, pprint

import conv_fpga_layer
import bunch_fpga_layer

from collections import defaultdict



available_layers = {
	'Convolution' : conv_layer.conv_layer(),   
	'Conv2D' : conv_layer.conv_layer(),   
	'BiasAdd' : matop_layer.matop_layer('BiasAdd'), 
	'AddN' : matop_layer.matop_layer('Add'), 
	'Reshape' : reshape_layer.reshape_layer(),
	#'Scale' : scale_layer.scale_layer(),    
    'Convolution' : conv_layer.conv_layer(),   
    'Conv2D' : conv_layer.conv_layer(),   
    'BiasAdd' : matop_layer.matop_layer('BiasAdd'),
    'Reshape' : reshape_layer.reshape_layer(),
    #'Scale' : scale_layer.scale_layer(),    
    'Relu' : relu_layer.relu_layer(),
    'ReLU' : relu_layer.relu_layer(),      
    #'Pooling' : pool_layer.pool_layer(),
    'MaxPool' : pool_layer.pool_layer('MAX'),
    'AvgPool' : pool_layer.pool_layer('AVG'), 
    'Concat' : concat_layer.concat_layer(3),
    'ConcatV2' : concat_layer.concat_layer(3),
    #'BatchNorm' : batchnorm_layer.batchnorm_layer(),
    'BatchNormWithGlobalNormalization' : batchnorm_layer.batchnorm_layer(),
    'InnerProduct' : matop_layer.matop_layer('MatMul'),
    'MatMul' : matop_layer.matop_layer('MatMul'),
    'Identity' :  identity_layer.identity_layer(),
    'CheckNumerics' : identity_layer.identity_layer(),
    'Softmax' : softmax_layer.softmax_layer()
}


class net_tf :
  def __init__(self, sess, prep_nodes, output, custom=True):
    self.variables = {}
    self.custom = custom
    self.output = output
    ignore_nodes = self.compute_ignore_nodes(sess, prep_nodes)
    self.schedule = self.create_schedule(sess, ignore_nodes)
    self.xdnnEnv = xdnn_env.xdnn_env()

  def compute_ignore_nodes(self, sess, nodes) :
    ignore = set()
    stk = nodes[:]
    while len(stk) > 0 :
      node = stk.pop()
      ignore.add(node)
      tensor = sess.graph.get_tensor_by_name(node + ':0')
      for inp in tensor.op.inputs :
        if inp.op.name not in ignore :
          stk.append(inp.op.name)
    return ignore


  def create_schedule(self,session, ignore_nodes) :
    schedule = []
    create_nodes = False

    for node in session.graph.get_operations() :
      if node.type == 'Const' :
        self.variables[node.name] = tf.contrib.util.make_ndarray(node.get_attr('value'))
    for node in session.graph.get_operations() :
      if node.type == 'Identity' :
        key = node.inputs[0].name.split(":")[0]
        self.variables[node.name] = self.variables[key]

    for node in session.graph.get_operations() :
      #print "ANDBG create_schedule op: %s %s" % (node.type, node.name)
      if node.name not in ignore_nodes :
        inps = [inp.op.name for inp in node.inputs] 
        if node.type == 'Const' :
          continue
        if self.custom and node.type in available_layers:
          layer = copy.deepcopy(available_layers[node.type])
          schedule.append(layer.prepare_layer(node, inps, self.variables))
        else :
          schedule.append(self.get_default_layer(node, inps))
        #schedule.append(layer.set_params(layer_params, self.variables))
      if node.name == self.output:
        break

    return schedule

  def gen_bunch_fpga_schedule(self, recipe) :
    print("Prepare Bunch FPGA schedule with recipe:")
    pprint.pprint(recipe)
  

    newSchedule = [] # we will be building this 

    doFPGASubgraph=False #flag that we need to run the subgraph on FPGA
    
    bunchXdlfLayers=[]

    for i,ol in enumerate(self.schedule):
      origLayerName = ol.output
      if origLayerName == recipe['startFPGA']:
        # 'start' signal to quantize subgraph until 'end' signal
        if isinstance(ol, conv_layer.conv_layer):
          doFPGASubgraph = True
          l = bunch_fpga_layer.bunch_fpga_layer(\
              weights = ol.filter_weights,
              stride = ol.conv_stride, 
              padding = ol.padding,
              quantize_key = recipe['name2key'](origLayerName),
              xdnn_env = self.xdnnEnv)
          l.setInput(ol.inputs)
          newSchedule.append(l)
        else:
          raise NotImplementedError(\
            "Not yet Implemented to start with %s %s"\
             % (origLayerName, type(ol)))
 
      if doFPGASubgraph:
        bunchXdlfLayers.append(ol)
        if recipe['startFPGA']==recipe['endFPGA']:
          raise NotImplementedError(\
            "Give singleStep Flow!!!!, not this flow")
      else:
        if not ol.output==recipe['startFPGA']:
          newSchedule.append(ol)
      
      if recipe['startFPGA']!=recipe['endFPGA']:
        if origLayerName == recipe['endFPGA']:
          l=newSchedule.pop()
          l.setOutput(origLayerName)
          l.setShape(ol.shape)
          l.setBunchXdlfLayers(bunchXdlfLayers)
          newSchedule.append(l)
          doFPGASubgraph=False

    # update schedule with new quantized schedule
    self.schedule = newSchedule

    for i,ol in enumerate(self.schedule):
      print(ol.inputs, ol.output, type(ol))
    return newSchedule

  def gen_fpga_schedule(self, recipe) :
    #Example recipe
    #  FPGA_Emulation_Quantize_Recipe = {
    #    "startFPGA": "conv1_7x7_s2/Conv2D",
    #    "endFPGA": "conv1_7x7_s2/Conv2D",
    #    "name2key": TFlayerName2QuantizeKey
    #     }

    print("Prepare FPGA schedule with recipe:")
    pprint.pprint(recipe)
  
    is_var_FPGA = {}
    def FPGA_var(x):
       if x in is_var_FPGA: 
         return x + "_FPGA"
       return x

    newSchedule = [] # we will be building this 

    doFPGASubgraph=False #flag that we need to run the subgraph on FPGA

    for i,ol in enumerate(self.schedule):
      origLayerName = ol.output
      if origLayerName == recipe['startFPGA']:
        # 'start' signal to quantize subgraph until 'end' signal
        doFPGASubgraph = True

      if doFPGASubgraph:
        if isinstance(ol, conv_layer.conv_layer):
          l = conv_fpga_layer.conv_fpga_layer(\
              weights = ol.filter_weights,
              stride = ol.conv_stride, 
              padding = ol.padding,
              quantize_key = recipe['name2key'](origLayerName),
              xdnn_env = self.xdnnEnv)
          if recipe['startFPGA']==recipe['endFPGA']:
            print("signal end FPGA")
            doFPGASubgraph=False
        else:
          raise NotImplementedError(\
            "Not yet Implemented %s %s" \
            % (origLayerName, type(ol)))
       # reroute to used FPGA vars, then add to new schedule
        #is_var_FPGA[origLayerName] = True
        #l.setInput(map(FPGA_var,ol.inputs))
        #l.setOutput(FPGA_var(origLayerName))
        l.setInput(ol.inputs)
        l.setOutput(origLayerName)
        l.setShape(ol.shape)
        newSchedule.append(l)
      else:
        if not ol.output==recipe['startFPGA']:
          newSchedule.append(ol)
      
      if recipe['startFPGA']!=recipe['endFPGA']:
        if origLayerName == recipe['endFPGA']:
          doFPGASubgraph=False

    # update schedule with new quantized schedule
    self.schedule = newSchedule

    return newSchedule


  def quantize_bunch_schedule(self, recipe) :
    #
    # Perform surgery on original schedule to quantize subgraph
    #
    # Example "recipe":
    # quantize_recipe = {
    #  "start": "conv1_7x7_s2/Conv2D",
    #  "end": "inception_5b_output",
    #  "quantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
    #  "unquantize": {"inception_5b_output":"inception_5b/pool_proj"},
    #  "name2key": layerName2QuantizeKey
    # }
    #
    print("Quantize schedule with recipe:")
    pprint.pprint(recipe)
  
    is_var_quantized = {}
    def quantized_var(x):
      if x in is_var_quantized: 
        return x + "_quantized"
      return x
    flag=0
    newSchedule = [] # we will be building this 
    quantizedLayers = defaultdict(int) # for stats
    doQuantizeSubgraph = False # flag that we need to quantize the subgraph

    for i,ol in enumerate(self.schedule):
      origLayerName = ol.output

      if origLayerName == recipe['start']:
        # 'start' signal to quantize subgraph until 'end' signal
        doQuantizeSubgraph = True

      if origLayerName in recipe['quantize']:
        # inject quantize_layer
        print("Start quantize subgraph @ %s" % origLayerName)
        quantizeKey = recipe['quantize'][origLayerName]
        l = quantize_layer.quantize_layer(
          quantizeKey, self.xdnnEnv)
        is_var_quantized[ol.inputs[0]] = True
        l.setInput([ol.inputs[0]])
        l.setOutput(quantized_var(ol.inputs[0]))

        newSchedule.append(l)

      if doQuantizeSubgraph:
        # substitute layers in quantized subgraph
        if isinstance(ol, conv_layer.conv_layer):
          l = bunch_hwemu_layer.bunch_hwemu_layer(\
            weights = ol.filter_weights,
            stride = ol.conv_stride, 
            padding = ol.padding,
            quantize_key = recipe['name2key'](origLayerName),
            xdnn_env = self.xdnnEnv)
        elif isinstance(ol, matop_layer.matop_layer) \
          and ol.optype == "BiasAdd":
          #print ol.output, origLayerName, 
          l = matop_hwemu_layer.matop_hwemu_layer(\
            ol.optype, ol.weights, ol.Bias,
            quantize_key = recipe['name2key'](origLayerName),
            xdnn_env = self.xdnnEnv)
        elif isinstance(ol, relu_layer.relu_layer):
          l = ol
        elif isinstance(ol, pool_layer.pool_layer):
          l = ol
        elif isinstance(ol, matop_layer.matop_layer) \
          and ol.optype == "Add":
          l = ol
        elif isinstance(ol, concat_layer.concat_layer) \
          or (hasattr(ol, "op") and "Concat" in ol.op.type):
          l = ol
        else: 
          raise NotImplementedError(\
            ":( don't know how to quantize %s %s" \
            % (origLayerName, type(ol)))

        quantizedLayers[type(ol)] += 1

        # reroute to used quantized vars, then add to new schedule

        if isinstance(ol ,matop_layer.matop_layer) \
          and ol.optype == "BiasAdd":
          l = newSchedule.pop()
          l.biasv3=ol.Bias
          newSchedule.append(l)
        elif isinstance(ol, relu_layer.relu_layer):
          l=newSchedule[-1]
          l1=ol
          print(type(ol.output), type(ol.inputs))
          l1.setInput([l.output])
          newSchedule.append(l1)
        else:
          is_var_quantized[origLayerName] = True
          l.setInput(list(map(quantized_var, ol.inputs)))
          l.setOutput(quantized_var(origLayerName))
          newSchedule.append(l)
      else:
        # add new schedule as-is
        newSchedule.append(ol)

      if origLayerName in recipe['unquantize']:
        # inject unquantize_layer
        print("End quantize subgraph @ %s" % origLayerName)
        quantizeKey = recipe['unquantize'][origLayerName]
        l = quantize_layer.unquantize_layer(
          quantizeKey, self.xdnnEnv)
        l.setInput([quantized_var(origLayerName)])
        l.setOutput(origLayerName)
        newSchedule.append(l)
        flag=1

      if origLayerName == recipe['end']:
        # 'end' signal to stop quantizing subgraph 
        doQuantizeSubgraph = False

    print("Quantized layers:")
    sortedQL = sorted(list(quantizedLayers.items()), key=operator.itemgetter(1))
    print(pprint.pprint(sortedQL))
    
    # update schedule with new quantized schedule
    self.schedule = newSchedule
    for i,ol in enumerate(newSchedule):
      print(ol.inputs, ol.output, type(ol))
    return newSchedule




  def quantize_schedule(self, recipe) :
    #
    # Perform surgery on original schedule to quantize subgraph
    #
    # Example "recipe":
    # quantize_recipe = {
    #  "start": "conv1_7x7_s2/Conv2D",
    #  "end": "inception_5b_output",
    #  "quantize": {"conv1_7x7_s2/Conv2D":"conv1/7x7_s2"},
    #  "unquantize": {"inception_5b_output":"inception_5b/pool_proj"},
    #  "name2key": layerName2QuantizeKey
    # }
    #
    print("Quantize schedule with recipe:")
    pprint.pprint(recipe)
  
    is_var_quantized = {}
    def quantized_var(x):
      if x in is_var_quantized: 
        return x + "_quantized"
      return x

    newSchedule = [] # we will be building this 
    quantizedLayers = defaultdict(int) # for stats
    doQuantizeSubgraph = False # flag that we need to quantize the subgraph

    for i,ol in enumerate(self.schedule):
      origLayerName = ol.output

      if origLayerName == recipe['start']:
        # 'start' signal to quantize subgraph until 'end' signal
        doQuantizeSubgraph = True

      if origLayerName in recipe['quantize']:
        # inject quantize_layer
        print("Start quantize subgraph @ %s" % origLayerName)
        quantizeKey = recipe['quantize'][origLayerName]
        l = quantize_layer.quantize_layer(
          quantizeKey, self.xdnnEnv)
        is_var_quantized[ol.inputs[0]] = True
        l.setInput([ol.inputs[0]])
        l.setOutput(quantized_var(ol.inputs[0]))

        newSchedule.append(l)

      if doQuantizeSubgraph:
        # substitute layers in quantized subgraph
        if isinstance(ol, conv_layer.conv_layer):
          l = conv_hwemu_layer.conv_hwemu_layer(\
            weights = ol.filter_weights,
            stride = ol.conv_stride, 
            activation = ol.activation_fn,
            padding = ol.padding,
            biases = ol.biases,
            mode = ol.mode,
            quantize_key = recipe['name2key'](origLayerName),
            xdnn_env = self.xdnnEnv)
        elif isinstance(ol, matop_layer.matop_layer) \
          and ol.optype == "BiasAdd":
          #print ol.output, origLayerName, 
          l = matop_hwemu_layer.matop_hwemu_layer(\
            ol.optype, ol.weights, ol.Bias,
            quantize_key = recipe['name2key'](origLayerName),
            xdnn_env = self.xdnnEnv)
        elif isinstance(ol, relu_layer.relu_layer):
          l = ol
        elif isinstance(ol, pool_layer.pool_layer):
          l = ol
        elif isinstance(ol, matop_layer.matop_layer) \
          and ol.optype == "Add":
          l = ol
        elif isinstance(ol, concat_layer.concat_layer) \
          or (hasattr(ol, "op") and "Concat" in ol.op.type):
          l = ol
        else: 
          raise NotImplementedError(\
            ":( don't know how to quantize %s %s" \
            % (origLayerName, type(ol)))

        quantizedLayers[type(ol)] += 1

        # reroute to used quantized vars, then add to new schedule
        is_var_quantized[origLayerName] = True
        l.setInput(list(map(quantized_var, ol.inputs)))
        l.setOutput(quantized_var(origLayerName))
        newSchedule.append(l)
      else:
        # add new schedule as-is
        newSchedule.append(ol)

      if origLayerName in recipe['unquantize']:
        # inject unquantize_layer
        print("End quantize subgraph @ %s" % origLayerName)
        quantizeKey = recipe['unquantize'][origLayerName]
        l = quantize_layer.unquantize_layer(
          quantizeKey, self.xdnnEnv)
        l.setInput([quantized_var(origLayerName)])
        l.setOutput(origLayerName)
        newSchedule.append(l)

      if origLayerName == recipe['end']:
        # 'end' signal to stop quantizing subgraph 
        doQuantizeSubgraph = False

    print("Quantized layers:")
    sortedQL = sorted(list(quantizedLayers.items()), key=operator.itemgetter(1))
    print(pprint.pprint(sortedQL))

    # update schedule with new quantized schedule
    self.schedule = newSchedule

    return newSchedule

  def compute_quantization(self, bitwidth, name2key, sess, input_dict, out):
    # save orig schedule -- we'll be modifying it for quantization
    origSchedule = self.schedule

    import xfdnn.tools.quantize.quantize_base as quant
    quantizer = quant.QuantParam()
    quantLayers = []

    # modify schedule to inject quantize_instrument_layers
    newSchedule = [] # we will be building this 
    for i,ol in enumerate(self.schedule):
      newSchedule.append(ol)

      # add instrumentation layers
      if isinstance(ol, conv_layer.conv_layer):
        if ol.inputs[0] == ol.output:
          raise NotImplementedError(\
            ":( don't know how to process in-place layer yet %s %s" \
            % (ol.output, type(ol)))

        quantizePrevKey = name2key(ol.inputs[0])
        quantizeKey = name2key(ol.output)
        quantLayers.append((quantizeKey, "unknown"))
        l = quantize_instrument_layer.quantize_instrument_conv_layer(\
          quantizer, bitwidth, [quantizePrevKey, quantizeKey], 
          ol.filter_weights)
        inputs = [ol.inputs[0], ol.output]
        l.setInput(inputs)
        l.setOutput(ol.output + "_quant_instrument")
        newSchedule.append(l)
      elif isinstance(ol, concat_layer.concat_layer) \
        or (hasattr(ol, "op") and "Concat" in ol.op.type):
        quantize_keys = [name2key(x) for x in ol.inputs]
        quantize_keys.append(name2key(ol.output))
        l = quantize_instrument_layer.quantize_instrument_concat_layer(\
          quantizer, bitwidth, quantize_keys)
        inputs = []
        inputs.extend(ol.inputs)
        inputs.append(ol.output)
        l.setInput(inputs)
        l.setOutput(ol.output + "_quant_instrument")
        newSchedule.append(l)

    # run with instrumented schedule
    self.schedule = newSchedule
    self.feed_forward(sess, input_dict, out)

    # save quantization result 
    quantizer.saveToJson(quantLayers, "sf_quantize.json")

    # restore orig schedule
    self.schedule = origSchedule

  def get_default_layer(self, node, inps) :
    l = layer_tf.layer_tf(inps, node.name, node.graph)
    return l

  def feed_forward(self, sess, input_dict, out):
    #self.variables[self.PreProcessEnd] = sess.run(sess.graph.get_tensor_by_name(self.PreProcessEnd+':0'), input_dict)
    for layer in self.schedule : 
      for k, v in list(input_dict.items()) :
        self.variables[k] = v

    for node in self.schedule :
      inputs = []
      for inp in node.inputs :
        if type(self.variables[inp]) == type(None) :
          print(inp, 'not found in variables')
        inputs.append(self.variables[inp])
    
      # ANDBG
      #if len(inputs[0].shape) == 4:
      #  print "\n"
      #  print type(node)
      #  print node.output + " input ==========================================="
      #  caffeInputs = np.transpose(inputs[0], [0,3,1,2])
      #  print caffeInputs.shape
      #  print caffeInputs
      #  print "\n"
    
      output = node.forward_exec(inputs)
      # ANDBG 
      #if len(output.shape) == 4:
      #  print "\n"
      #  print node.output + " output ==========================================="
      #  caffeOutput = np.transpose(output, [0,3,1,2])
      #  print caffeOutput.shape
      #  print caffeOutput
      #  print "\n"

      self.variables[node.output] = output
      #print 'CDBG:', node.inputs, type(node), node.output, self.variables[node.output].shape
    return self.variables[out]
