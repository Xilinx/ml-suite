#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from collections import OrderedDict
from copy import deepcopy
from six import string_types as _string_types

import tensorflow as tf
import numpy as np
from tensorflow.python.ops import script_ops as _script_ops

from xfdnn.tools.compile.bin.xfdnn_compiler_tensorflow import TFFrontend
from xfdnn.rt import xdnn_util
from xfdnn.rt import xdnn_util_tf
from xfdnn.rt.xdnn_rt_base import xdnnRT as _xdnnRT
from xfdnn.rt.xdnn_opt import CPUTransform, HWEmuTransform, FPGATransform
from ext.PyTurboJPEG import imread as _imread



## global variables
global_fpga_device    = 'cpu:0'   ## TODO: replace with FPGA:0
global_pyfunc_counter = 0

######################################################
## tensorflow specific utility functions
######################################################
## expanding tf.NodeDef methods
tf.NodeDef.set_name  = xdnn_util_tf.set_name
tf.NodeDef.set_shape = xdnn_util_tf.set_shape
tf.NodeDef.get_shape = xdnn_util_tf.get_shape
tf.NodeDef.get_dtype = xdnn_util_tf.get_dtype

## expanding tf.GraphDef methods
tf.GraphDef.get_node_dict   = xdnn_util_tf.get_node_dict
tf.GraphDef.get_output_dict = xdnn_util_tf.get_output_dict
tf.GraphDef.get_node_index  = xdnn_util_tf.get_node_index
tf.GraphDef.remove_nodes    = xdnn_util_tf.remove_nodes
tf.GraphDef.is_cyclic       = xdnn_util_tf.is_cyclic
tf.GraphDef.all_cycles      = xdnn_util_tf.all_cycles




class TFxdnnRT(_xdnnRT):
    def __init__ (self, args, **kwargs):
        super(TFxdnnRT, self).__init__(TFFrontend, args, **kwargs)

        if not hasattr(self, 'graph_def'):
          self.graph_def = deepcopy(self._graph)

        if not hasattr(self, 'fpga_pynode_dict'):
          self.fpga_pynode_dict = OrderedDict()

    def load_graph(self, args, **kwargs):
        return xdnn_util_tf.load_graph(args, **kwargs)

    def list_inputs_of_graph(self, graph_def):
        return xdnn_util_tf.discover_sourcenodes(graph_def)

    def list_outputs_of_graph(self, graph_def):
        if hasattr(self, 'outputs') and self.outputs is not None:
          return self.outputs
        return xdnn_util_tf.discover_sinknodes(graph_def)

    def extract_subgraph(self, outputs, inputs, inclusive=False, filename=None, session=None):
        destgraph_def = xdnn_util_tf.extract_subgraph(self._graph, outputs, inputs, inclusive)
        destgraph_def, fValidGraph = xdnn_util_tf.freeze_graph(session,
                                                               destgraph_def,
                                                               sinknodes_list=outputs,
                                                               filename=filename,
                                                               freeze_blacklist=[],
                                                               freeze_whitelist=[])
        return destgraph_def

    def device_transforms(self, args):
        subTFs = []
        for partition in self.graph_partitions:
          ## extract supported subgraph
          if partition.supported:
            print('Re-compile partition_index \"{}\"'.format(partition.index))

            name_postfix = '_partition#{:02d}'.format(partition.index)
            filename = self.file_path('.pb', name_postfix=name_postfix)

            subTFs.append((TFxdnnRT(args,
                                    fPartition=False,
                                    startnode=partition.inputMap.values(),
                                    finalnode=partition.outputMap.values(),
                                    inclusive=False,
                                    name_postfix=name_postfix,
                                    weights=filename,
                                    graph_savepath=filename,
                                    cpulayermustgo=True,
                                   ),
                           filename,
                           partition.index
                          )
                         )
          else:
            partition.layers    = list(partition.names)
            partition.variables = {}

        for TF, filename, partition_index in subTFs:
          partition = self.graph_partitions[partition_index]

          print('Transorm partition_index \"{}\"'.format(partition.index))

          time_to_layer_list = []
          for time, layers in TF.compilerSchedule.time_to_layer.items():
            for layer in layers:
              time_to_layer_list.append((time, layer))

          ## NOTE: hack to get rid of input placeholder
          ## FIXME: this might cause problem if TF.compilerJson['inputs'] info is used in
          ##        FPGA. Commented out for now when compiler flag "cpulayermustgo" is used.
          ## ALTERNATIVE?: time_to_layer_list = [tol for tol in time_to_layer_list if not tol[1].startswith('geph__')]
          # time_to_layer_list = [tol for tol in time_to_layer_list if tol[1] not in set(TF.inputs)]

          if args.device.lower() == "cpu":
            opt = CPUTransform(time_to_layer_list, TF.layerparameter_dict, args, TF._graph)
          elif args.device.lower() == "hwemu":
            opt = HWEmuTransform(partition.inputs, TF.pydotGraph, TF.compilerSchedule, args)
          elif args.device.lower() == "fpga":
            if not args.fpga_recipe:
              args.fpga_recipe = {'start': [inp['input_name'] for inp in
                                            TF.compilerJson['inputs']],
                                  'end':   [out['output_name'] for out in
                                            TF.compilerJson['outputs']]}
            if args.xclbin:
              opt = FPGATransform(time_to_layer_list, TF.layerparameter_dict,
                                  TF.compilerJson, args, TF._graph,
                                  filename=filename)
            else:
              raise AttributeError("Must specify path to xclbin when device = FPGA")
          else:
            raise AttributeError("Unsupported device type", args.device)

          #variables hold the inputs/consts of graph
          partition.layers    = opt.getLayers()
          partition.variables = opt.variables
          for l in partition.layers:
            l.setup()

    def rebuild_graph(self):
        self.graph_def = self.rebuild_graph_def()
        # if len(self.graph_partitions) > 0:
        #   self.graph_def = self.rebuild_graph_def()

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
        #    ## NOTE: This doesn't work if the supported partition is not maximal
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
        def insert_fpga_pynode(graph_def, partition):
            global global_pyfunc_counter

            # def update_partition_boundries(partition, graph_output_dict):
            #     for extra, layer in partition.spt_set.items():
            #       if layer not in partition.names:
            #         raise RuntimeError('supported layer {} not in partition {}-{}'.format(layer,
            #                                                                               partition.index,
            #                                                                               partition.names))
            #       for output in graph_output_dict[extra].keys():
            #         if output not in partition.spt_set and layer not in partition.outputs:
            #           partition.outputs += [layer]
            #           partition.outputMap.update({layer: extra})


            graph_node_dict, \
            graph_output_dict   = graph_def.get_node_dict(outmap=True)

            input_partitions    = self.reverse_connectivity[partition.index]
            consumer_partitions = self.connectivity[partition.index]

            input_tensors           = []
            placeholder_replace_map = {}

            # update_partition_boundries(partition, graph_output_dict)

            with tf.Graph().as_default() as fpga_subgraph:
              for idx, inp in enumerate(partition.inputs):
                ## NOTE: index of partition in self.graph_partitions must match partition.index
                input_partition = input_partitions[inp]
                if len(set(input_partition)) > 1:
                  raise RuntimeError('input \"{}\" cannot belong to multiple partitions! {}'.format(inp, set(input_partition)))
                if not self.graph_partitions[input_partition[0]].supported:
                  inp = partition.inputs[idx] = partition.inputMap[inp]

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

                  # print('_names_in_use: ', fpga_subgraph._names_in_use)

                  output_shapes = [graph_node_dict[output].get_shape() for output in partition.outputs]
                  output_dtypes = [graph_node_dict[output].get_dtype() for output in partition.outputs]

                  if self.data_format != 'NCHW':
                    with fpga_subgraph.name_scope('fpga_preproc'):
                      fpga_input_tensors = [tf.transpose(tensor, [0, 3, 1, 2]) if
                                            len(tensor.get_shape()) == 4 else tensor for
                                            tensor in input_tensors]
                  else:
                    fpga_input_tensors = input_tensors

                  # with tf.device(global_fpga_device):
                  fpga_output_tensors = tf.py_func(partition.forward_exec,
                                                   fpga_input_tensors,
                                                   output_dtypes,
                                                   stateful=False)

                  for fpga_output_tensor, output_shape in zip(fpga_output_tensors, output_shapes):
                    if len(output_shape) == 4:
                      fpga_output_tensor.set_shape([output_shape[i] for i in (0, 3, 1, 2)])
                    else:
                      fpga_output_tensor.set_shape(output_shape)

                  # print('graph py_func tokens: {}'.format(_script_ops._py_funcs._funcs.keys()))

                  if self.data_format != 'NCHW':
                    with fpga_subgraph.name_scope('fpga_postproc'):
                      output_tensors = [tf.transpose(tensor, [0, 2, 3, 1]) if
                                        len(tensor.get_shape()) == 4 else tensor for tensor in
                                        fpga_output_tensors]
                  else:
                    output_tensors = fpga_output_tensors

                  # print('graph py_func tokens: {}'.format(_script_ops._py_funcs._funcs.keys()))

                  global_pyfunc_counter += 1

              ## create identity nodes matching the correct output names
              # /wrk/hdstaff/arminb/Anaconda/envs/deephi_tf/lib/python2.7/site-packages/tensorflow/python/framework/op_def_library.py:394
              # /wrk/hdstaff/arminb/Anaconda/envs/deephi_tf/lib/python2.7/site-packages/tensorflow/python/framework/ops.py:6007
              # /wrk/hdstaff/arminb/Anaconda/envs/deephi_tf/lib/python2.7/site-packages/tensorflow/python/framework/ops.py:4117
              output_tensors = [tf.identity(tensor, name=partition.outputMap[name]) for tensor, name
                                in zip(output_tensors, partition.outputs)]

            ## maintain an ordered dictionary of pynodes for loading the graph
            #self.fpga_pynode_dict[fpga_output_tensors[0].op.node_def.attr['token'].s] = partition
            self.fpga_pynode_dict[fpga_output_tensors[0].op.name] = (partition, [inp.name for inp in
                                                                             fpga_input_tensors],
                                                                     list(output_dtypes))

            fpga_subgraph_def    = fpga_subgraph.as_graph_def(add_shapes=True)
            # fpga_node_map, \
            # fpga_output_node_map = fpga_subgraph_def.get_node_dict(outmap=True)

            for key, value in placeholder_replace_map.items():
              if key != value:
                raise RuntimeError('real and dummy placeholder names mismatched! {}'.format(placeholder_replace_map))

            ##########################################
            ## NOTE: NOT needed anymore (identity node fixes name mismatches)
            ##########################################
            # ## remove dummy placehoders and connect their consumers to the original inputs
            # fpga_nodes = []
            # for fpga_node_name, fpga_node in fpga_node_map.items():
            #   if fpga_node.op == 'Placeholder':
            #     # replace palceholder with original inputs
            #     placeholder_consumers = fpga_output_node_map[fpga_node_name]
            #     for placeholder_consumer_name, input_index in placeholder_consumers.items():
            #       placeholder_consumer_node = fpga_node_map[placeholder_consumer_name]
            #       del placeholder_consumer_node.input[input_index]
            #       placeholder_consumer_node.input.insert(input_index, placeholder_replace_map[fpga_node_name])
            #   else:
            #     # All nodes except dummy placeholders are to be copied to the main graph
            #     fpga_nodes += [fpga_node]

            # ## connect input of fpga_consumer_nodes to output from py_function
            # for i, outp in enumerate(partition.outputs):
            #   #output_consumer_nodes = {}
            #   #discover_consumer_nodes((outp, -1), set(partition.outputs), output_consumer_nodes)
            #   output_consumer_nodes = discover_consumer_nodes(partition.outputMap[outp],
            #                                                   set(partition.names), graph_output_dict)

            #   ## update the input list for the consumers partitions of outp
            #   for consumerPart in consumer_partitions[outp]:
            #     consumerPart = self.graph_partitions[consumerPart]
            #     # print(outp, output_consumer_nodes, consumerPart.index, consumerPart.supported)
            #     if consumerPart.supported:
            #       temp = self.reverse_connectivity[consumerPart.index]
            #       temp[output_tensors[i].name] = temp[outp]
            #       temp.pop(outp)
            #       consumerPart.inputs = [name if name != outp else output_tensors[i].name for name in consumerPart.inputs]
            #       consumerPart.inputMap[output_tensors[i].name] = consumerPart.inputMap[outp]

            #   ## update the input list for the consumer nodes of outp
            #   for output_consumer_node_name, input_index in output_consumer_nodes.items():
            #     output_consumer_node = graph_node_dict[output_consumer_node_name]
            #     del output_consumer_node.input[input_index]
            #     output_consumer_node.input.insert(input_index, output_tensors[i].name)

            ## remove dummy placeholders
            fpga_subgraph_def.remove_nodes([inp.op.name for inp in input_tensors])

            ## remove original output from input graph
            graph_def.remove_nodes([out.op.name for out in output_tensors])

            ## append fpga nodes to original graph
            fpga_nodes = fpga_subgraph_def.node
            graph_def.node.extend(fpga_nodes)

            return graph_def, fpga_nodes

        #############################
        ## function's body
        #############################
        graph_def = deepcopy(self._graph)

        self.fpga_pynode_dict = OrderedDict()

        for t, partition in enumerate(self.graph_partitions):
          if partition.supported:
            graph_def, __ = insert_fpga_pynode(graph_def, partition)

        # check graph integrity
        isCyclic, cycle = graph_def.is_cyclic()
        if isCyclic:
          raise RuntimeError('Graph partitioning resulted in cyclic graph; {}'.format(graph_def.all_cycles()))

        ## To make sure the modified graph_def is valid
        #with tf.Graph().as_default():
        #  tf.import_graph_def(graph_def, name='')

        # cleanup graph (at this point we do not have variables in graph so no initialization (sess) is needed)
        graph_def, fValidGraph = xdnn_util_tf.freeze_graph(None,
                                                           graph_def,
                                                           sinknodes_list=self.outputs,
                                                           remove_training_nodes=False,
                                                           filename=self.file_path('.pb', name_postfix='-fpga'),
                                                          )

        return graph_def

    def load_partitioned_graph(self):
        ## import the base partitioned graph
        tf.reset_default_graph()
        tf.import_graph_def(self.graph_def, name='')
        graph = tf.get_default_graph()

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

        return graph


    def forward_exec(self, **kwargs):
        def default_forward_exec(**kwargs):
            if 'sess' in kwargs:
              sess  = kwargs['sess']
              graph = sess.graph
            else:
              #config=tf.ConfigProto(log_device_placement=True)
              graph = self.load_partitioned_graph()
              sess  = tf.Session(graph=graph, config=kwargs.get('config', None))

            ## declare input tensors to network graph
            input_names = kwargs.get('input_names', None)
            if not input_names:
              input_names = self.inputs
            else:
              for inp in input_names:
                if not isinstance(inp, _string_types):
                  raise TypeError('input_names should be flattened list of name strings')
            input_tensors  = [graph.get_operation_by_name(name).outputs[0] for name in input_names]

            ## declare output tensors to network graph
            output_names = kwargs.get('output_names', None)
            if not output_names:
              output_names = self.outputs
            else:
              for out in output_names:
                if not isinstance(out, _string_types):
                  raise TypeError('output_names should be flattened list of name strings')
            output_tensors = [graph.get_operation_by_name(name).outputs[0] for name in output_names]

            ## bound the input tensors to input data
            preprocess   = kwargs.get('preprocess', self.preprocess)
            input_values = xdnn_util.make_list(kwargs.get('input_values', None))

            feed_dict = {inp_tensor: inp_val if len(inp_val.shape) == len(inp_tensor.shape) else
                         inp_val[None,...] for inp_tensor, inp_val in zip(input_tensors, preprocess(input_values))}

            output_values = sess.run(output_tensors, feed_dict=feed_dict if feed_dict else None)

            if 'sess' not in kwargs:
              sess.close()

            return {tensor.op.name: val for val, tensor in zip(output_values, output_tensors)}

        if 'forward_exec' in kwargs:
          return kwargs['forward_exec'](*kwargs.get('argv', None))
        else:
          return default_forward_exec(**kwargs)

    def preprocess(self, inputs):
        ## inputs should be a list
        if not isinstance(inputs, list):
          raise TypeError('inputs to preprocessing should be a list')

        res = []
        for inp in inputs:
          if isinstance(inp, np.ndarray):
            res.append(inp)
          elif isinstance(inp, _string_types):
            res.append(_imread(inp))
        return res


    def debug_finalnode(self, input_values, **kwargs):
        input_values = xdnn_util.make_list(input_values)

        input_names  = self.inputs
        output_names = xdnn_util.make_list(self._args.finalnode)

        ## running the original graph
        with tf.Graph().as_default() as graph_org:
          tf.import_graph_def(self._graph, name='')

          node_dict, \
          output_dict    = graph_org.as_graph_def().get_node_dict(outmap=True)
          input_tensors  = [graph_org.get_operation_by_name(name).outputs[0] for name in input_names]
          output_tensors = [graph_org.get_operation_by_name(name).outputs[0] for name in output_names]

          print('org tensor: {}'.format(output_tensors))
          with tf.Session() as sess:
            feed_dict = {tensor: [value] for tensor, value in zip(input_tensors, input_values)}
            ret = {'org': sess.run(output_tensors, feed_dict)}


        ## running the partitioned graph
        graph          = self.load_partitioned_graph()
        input_tensors  = [graph.get_operation_by_name(name).outputs[0] for name in input_names]
        output_tensors = [graph.get_operation_by_name(name).outputs[0] for name in output_names]

        # output_tensors = []
        # for name in output_names:
        #   consumers = output_dict[name].keys()
        #   name = consumers[0] if consumers else name
        #   inp = [node.name for node in graph.get_operation_by_name(name).inputs if 'transpose' in
        #          node.name]
        #   if inp:
        #     ## use this in case output_names are NOT the last operations in global graph
        #     output_tensors.append(inp[0])
        #   else:
        #     ## use this in case output_names are the last operations in global graph
        #     output_tensors.append(graph.get_operation_by_name(name).outputs[0])
        ##########################################

        print('fpga tensor: {}'.format(output_tensors))
        with tf.Session(graph=graph) as sess:
          feed_dict = {tensor: [value] for tensor, value in zip(input_tensors, input_values)}
          ret.update({'fpga': sess.run(output_tensors, feed_dict)})

        for org, fpga in zip(ret['org'], ret['fpga']):
          print('org  max {} min {} mean {} std {}'.format(org.max(), org.min(), org.mean(), org.std()))
          print('FPGA max {} min {} mean {} std {}'.format(fpga.max(), fpga.min(), fpga.mean(), fpga.std()))
          print('DIFF max {} min {} mean {} std {}'.format((fpga-org).max(), (fpga-org).min(),
                                                           (fpga-org).mean(), (fpga-org).std()))

        return ret


#    class GEMXxdnnRT(TFxdnnRT):
#        def __init__(self, compilerFunc, args, **kwargs):
#            args = xdnn_util.dict2attr(args)
#            args.update(kwargs)
#
#            if args.networkfile is None and args.loadpickle is None:
#              raise AttributeError('network argument is missing')
#
#            self.fPartitioned  = args.get('fPartition', True)
#            self.networkfile   = args.networkfile
#            self.data_format   = args.data_format
#            self.save2modeldir = args.save2modeldir   # whether to save to mo directory
#
#            name_postfix       = args.get('name_postfix', '_frozen')
#            self.picklefile    = self.file_path('.pickle', name_postfix=name_postfix)   # save pydotGraph and compilerJson
#
#            ## run compiler
#            compiler = compilerFunc(args,
#                                    savepickle=self.picklefile,
#                                    loadpickle=self.picklefile,
#                                   )
#
#            self._graph, \
#            self.inputs, \
#            self.outputs, \
#            pydotGraph, \
#            compilerSchedule, \
#            ssize, \
#            compilerJson \
#                = compiler.compile()
#
#            if compilerJson is None:
#              raise RuntimeError('Compiler failed to produce valid schedule')
#
#            ## store compiler json
#            with open(self.file_path('.json', name_postfix=name_postfix+'-xdlfCompiler'), "w") as f:
#              json.dump(compilerJson, f, sort_keys=True, indent=4, separators=(',',': '))
#
#            if (self._graph  is None or
#                self.inputs  is None or
#                self.outputs is None):
#              ## in case loading from a pickled file
#              self._graph, \
#              self.inputs, \
#              self.outputs, \
#              self.fValidGraph \
#                  = self.load_graph(args)
#
#            self.unspt_set      = set(compilerJson['unsupported']['list'].keys()) # build unsupported layer set
#            layerparameter_dict = {}
#            layeroutput_dict    = defaultdict(lambda: {})
#            for node in pydotGraph.get_nodes():
#              LayerParameter = node.get('LayerParameter')
#              layer_name = LayerParameter.name
#              layerparameter_dict[layer_name] = LayerParameter
#              if 'blob' not in layer_name and LayerParameter.bottoms:       ## FIXME: hack to remove output blobs from output list
#                for input_index, input_name in enumerate(LayerParameter.bottoms):
#                  layeroutput_dict[input_name][layer_name] = input_index
#
#            self.layerparameter_dict = layerparameter_dict
#            self.pydotGraph = pydotGraph
#            self.compilerSchedule = compilerSchedule
#            self.compilerJson = compilerJson
#
#            graph_partitions = [LayerPartition(0, False),
#                                LayerPartition(1, True),
#                                LayerPartition(2, False),
#                                LayerPartition(1, False)]
#
#            ################## now manually add inputs and outputs for all paritions
#            graph_partitions[0].inputs = ['',]
#            graph_partitions[0].outpus = ['',]
#            graph_partitions[1].inputs = ['',]
#            graph_partitions[1].outpus = ['',]
#            graph_partitions[2].inputs = ['',]
#            graph_partitions[2].outpus = ['',]
#            graph_partitions[3].inputs = ['',]
#            graph_partitions[3].outpus = ['',]
#
#            for partition in graph_partitions:
#              partition.inputMap = {name: name for name in partition.inputs}
#              partition.outputMap = {name: name for name in partition.outputs}
#            ########################################################################
#
#            ## optimize partitions
#            self.graph_partitions = self.refine_graph_partitions(graph_partitions)
#
#            ## device transformation for supported layers
#            ################## now need to continue the device_transforms function below to plug in your
#            ################## own function. your function should support opt.getLayers() and
#            ################## opt.variables.
#            ################## layers from opt.getLayers() should support layer.forward_exec(input)
#            ################## methods. follow xdnn_opt and py files in xfdnn/tools/emu.
#            self.device_transforms(args)
#            ########################################################################
#
#            self.rebuild_graph()
#
#        def device_transforms(self, args):
#            subTFs = []
#            for partition in self.graph_partitions:
#              ## extract supported subgraph
#              print('Transorm partition_index \"{}\"'.format(partition.index))
#              print('inputs:  {}'.format(partition.inputMap.keys()))
#              print('actual:  {}'.format(partition.inputMap.values()))
#              print('outputs: {}'.format(partition.outputMap.keys()))
#              print('actual:  {}'.format(partition.outputMap.values()))
#
#              if partition.supported:
#                opt = GEMXTransform(.....)
#
#                partition.layers    = opt.getLayers()
#                partition.variables = opt.variables
