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
import operator, pprint
import pydot

import layer
import conv_layer
import conv_hwemu_layer
import concat_layer
import identity_layer
import pool_layer
import reshape_layer
import matop_layer
import matop_hwemu_layer
import quantize_layer
import softmax_layer
import relu_layer
import batchnorm_layer
import scale_layer
import layer_tf
import reduce_layer


import tensor_tools as tt
import keras_tools as kt
import layer_tf

import xdnn_env
from collections import defaultdict

available_layers = {
    'Convolution' : conv_layer.conv_layer(mode = 'NCHW'),   # done
    'BiasAdd' : matop_layer.matop_layer('BiasAdd'), #done
    'Eltwise' : matop_layer.matop_layer('Add'), # TODO FIXME assumes add???
    #'Mean' : reduce_layer.reduce_layer('AVG', mode='NCHW'), # TODO FIXME assumes avgpool???
    'Reshape' : reshape_layer.reshape_layer(),
    'Scale' : scale_layer.scale_layer(),    #done
    'ReLU' : relu_layer.relu_layer(),    #done  
    'Pooling' : pool_layer.pool_layer(mode='NCHW'),     #done
    'Concat' : concat_layer.concat_layer(1), #done
    'BatchNorm' : batchnorm_layer.batchnorm_layer(),  #done
    'InnerProduct' : matop_layer.matop_layer('MatMul'), #done
    #'Mul' : matop_layer.matop_layer('MatMul'), #done
    'Sub' : matop_layer.matop_layer('Sub'), #done
    'Identity' :  identity_layer.identity_layer(),
    'Softmax' : softmax_layer.softmax_layer()

}

class net :

    def __init__(self, graph, schedule, prep_nodes, out, custom) :
        self.variables = {}
        self.constSet = set()
        self.custom = custom
        self.out = out
        #self.pydotG, sch = tt.from_tfgraph_xddgraph(session.graph,'outputpng.png')
        ignore = self.compute_ignore_nodes(prep_nodes, graph)
        self.schedule = self.create_schedule(graph, ignore, schedule, custom)
        self.xdnnEnv = xdnn_env.xdnn_env()
        

    def compute_ignore_nodes(self, nodes, graph) :
        ignore = set()
        stk = nodes[:]
        while len(stk) > 0 :
            node_name = stk.pop()
            ignore.add(node_name)
            g_node = graph.get_node(pydot.quote_if_necessary(node_name))
            if len(g_node) == 0 : continue
            g_node = g_node[0]
            params = g_node.get('LayerParameter')
            if params.bottoms == None : continue
            for inp in params.bottoms :
                if inp not in ignore :
                    stk.append(inp)
        return ignore

    def create_schedule(self, graph, ignore, sch, custom) :
        print("CBDG : Creating schedule")
        objmap = {}
        print('ignores :', ignore)
        for node in graph.get_nodes() :
            P = node.get('LayerParameter')
            objmap[P.name] = node
        schedule = []
        for k in range(len(sch)) :
            v = sch[k]
            #assuming that there is only one operation happening per time.
            node = objmap[v.active_node_names[0]]
            layer_params = node.get('LayerParameter')

            #print "\nANDBG create_schedule %s %s" % (layer_params.name, layer_params.type)
            #print layer_params # ANDBG
            #print layer_params.layer[0]

            if layer_params.type[0] == 'Const' :
                self.variables[layer_params.name] = layer_params.data
                self.constSet.add(layer_params.name)
                continue
            if layer_params.name not in ignore :
                print(layer_params.name, layer_params.type[0])
                if layer_params.type[0] in available_layers :
                    layer = copy.deepcopy(available_layers[layer_params.type[0]])
                    schedule.append(layer.set_params(layer_params, self.variables))
                else :
                    schedule.append(self.get_default_layer(layer_params))
            if layer_params.name == self.out :
                break
        return schedule
    
    def get_default_layer(self,layer_params) :
        print(layer_params, layer_params.layer[0].type, layer_params.layer[0].name)
        l = layer_tf.layer_tf(layer_params.bottoms, layer_params.tops[0], layer_params.layer[0].graph, 'NCHW')
        l.get_costant_inputs(self.constSet)
        return l

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
          #elif isinstance(ol, matop_layer.matop_layer) \
          #  and ol.optype == "BiasAdd":
          #  #print ol.output, origLayerName, 
          #  l = matop_hwemu_layer.matop_hwemu_layer(\
          #    ol.optype, ol.weights, ol.Bias,
          #    quantize_key = recipe['name2key'](origLayerName),
          #    xdnn_env = self.xdnnEnv)
          #elif isinstance(ol, relu_layer.relu_layer):
          #  l = ol
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


    def feed_forward(self, sess, input_dict, out):
        #self.variables[self.PreProcessEnd] = sess.run(sess.graph.get_tensor_by_name(self.PreProcessEnd+':0'), input_dict)
        for k, v in list(input_dict.items()) :
            self.variables[k] = v
        for node in self.schedule :
            inputs = []
            for inp in node.inputs :
                inputs.append(np.copy(self.variables[inp]))

            # ANDBG 
            #print "\n"
            #print node.output + " input ==========================================="
            #print inputs[0].shape
            #print inputs[0]
            #print "\n"
            '''output = None
            if type(node) != layer_tf.layer_tf :
                output = node.forward_exec(inputs)
            else :
                for i in range(len(inputs)) :
                    if type(inputs[i]) == np.ndarray and len(inputs[i].shape) == 4 and node.inputs[i] not in self.constSet :
                        inputs[i] = np.transpose(inputs[i], [0,2,3,1])
                output = node.forward_exec(inputs)
                if len(output.shape) == 4 :
                    output = np.transpose(output, [0,3,1,2])'''
            
            # ANDBG 
            #print "\n"
            #print node.output + " output ==========================================="
            #print output.shape
            #print output
            #print "\n"

            #print type(output)
            '''self.variables[node.output] = output'''
            self.variables[node.output] = node.forward_exec(inputs)
            #if node.output == 'softmax/logits' :
                #print output
        return self.variables[out]
