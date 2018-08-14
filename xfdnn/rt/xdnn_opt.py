import copy
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

import sys
import xdnn_env

class CPUTransform :

    def __init__(self, inputs, graph, schedule):
        self.variables = {}
        self.constSet = set()
        
        self.available_layers = {
            'Convolution' : conv_layer.conv_layer(mode='NCHW'),  # done
            'BiasAdd' : matop_layer.matop_layer('BiasAdd'),  # done
            'Eltwise' : matop_layer.matop_layer('Add'),  # TODO FIXME assumes add???
            # 'Mean' : reduce_layer.reduce_layer('AVG', mode='NCHW'), # TODO FIXME assumes avgpool???
            'Reshape' : reshape_layer.reshape_layer(),
            'Scale' : scale_layer.scale_layer(),  # done
            'ReLU' : relu_layer.relu_layer(),  # done  
            'Pooling' : pool_layer.pool_layer(mode='NCHW'),  # done
            'Concat' : concat_layer.concat_layer(1),  # done
            'BatchNorm' : batchnorm_layer.batchnorm_layer(),  # done
            'InnerProduct' : matop_layer.matop_layer('MatMul'),  # done
            # 'Mul' : matop_layer.matop_layer('MatMul'), #done
            'Sub' : matop_layer.matop_layer('Sub'),  # done
            'Identity' :  identity_layer.identity_layer(),
            'Softmax' : softmax_layer.softmax_layer()
        }

        ignore = self.compute_ignore_nodes(inputs, graph)
        self._layers = self.create_schedule(graph, ignore, schedule)

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

    def create_schedule(self, graph, ignore, sch) :
        print("CBDG : Creating schedule")
        objmap = {}
        print('ignores :', ignore)
        for node in graph.get_nodes() :
            P = node.get('LayerParameter')
            objmap[P.name] = node
        schedule = []
        for k in range(len(sch)) :
            v = sch[k]
            # assuming that there is only one operation happening per time.
            node = objmap[v.active_node_names[0]]
            layer_params = node.get('LayerParameter')

            print "\nANDBG create_schedule %s %s" % (layer_params.name, layer_params.type)
            print layer_params # ANDBG
            print layer_params.layer[0]

            if layer_params.type[0] == 'Const' :
                self.variables[layer_params.name] = layer_params.data
                self.constSet.add(layer_params.name)
            elif layer_params.name not in ignore :
                print(layer_params.name, layer_params.type[0])
                if layer_params.type[0] in self.available_layers :
                    layer = copy.deepcopy(self.available_layers[layer_params.type[0]])
                    layer.set_params(layer_params, self.variables)
                    schedule.append(layer)
                else :
                    schedule.append(self.get_default_layer(layer_params))
                    
        sys.exit()
        
        return schedule
    
    def get_default_layer(self, layer_params) :
        l = layer_tf.layer_tf(layer_params.bottoms, layer_params.tops[0], layer_params.layer[0].graph, 'NCHW')
        l.get_costant_inputs(self.constSet)
        return l
    
    def getLayers(self):
        return self._layers

        
class FPGATransform (CPUTransform):
    
    def __init__(self, inputs, graph, schedule, xclbin):
        CPUTransform.__init__(self,inputs, graph, schedule)
        self.xdnnEnv = xdnn_env.xdnn_fpga_env(xclbin)
        newSchedule = []  # we will be building this 
        for ol in self._layers:
            origLayerName = ol.output
    
            if isinstance(ol, conv_layer.conv_layer):
                l = conv_fpga_layer.conv_fpga_layer(\
                      weights=ol.filter_weights,
                      stride=ol.conv_stride,
                      padding=ol.padding,
                      quantize_key=self.TFlayerName2QuantizeKey(origLayerName),
                      xdnn_env=self.xdnnEnv)
                l.setInput(ol.inputs)
                l.setOutput(origLayerName)
                l.setShape(ol.shape)
                newSchedule.append(l)
            else:
                newSchedule.append(ol)
        # update schedule with new quantized schedule
        self._layers = newSchedule

    def TFlayerName2QuantizeKey(self,name):
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
        
class HWEmuTransform:
    pass
