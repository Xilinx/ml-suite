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

import layer
import tensorflow as tf
import numpy as np

class layer_tf(layer.layer):
    def __init__(self, inps = None, out = None, graph = None, mode='NHWC') :
        self.mode = mode
        if inps != None :
            self.setInput(inps)
        if out != None :
            self.setOutput(out)
        self.graph = graph
        self.consts = set()
        if inps != None and out != None and graph != None:
            self.graph = self.get_subgraph(graph)

    def get_costant_inputs(self, constSet) :
        for inp in self.inputs :
            if inp in constSet :
                self.consts.add(inp)

    def get_subgraph(self, graph) :
        graphdef = graph.as_graph_def()
        sub_g_def = tf.graph_util.extract_sub_graph(graphdef,[self.output])
        destgraph = None
        with tf.Graph().as_default() :
            tf.import_graph_def(sub_g_def, name = '')
            outops = [tf.get_default_graph().get_operation_by_name(self.output)]
            inops = [tf.get_default_graph().get_operation_by_name(inp) for inp in self.inputs]
            ops = tf.contrib.graph_editor.get_within_boundary_ops(tf.get_default_graph(), outops, inops)
            for inp in inops :
                ops.remove(inp)
            sub_g_view = tf.contrib.graph_editor.make_view(ops)
            inp_names_before = [inp.name for inp in sub_g_view.inputs]
            tf.contrib.graph_editor.detach_inputs(sub_g_view)
            inp_names_after = [inp.name for inp in sub_g_view.inputs]
            self.name_map = {inp_names_before[i] : inp_names_after[i].split('/')[-1] for i in range(len(inp_names_before))}
            destgraph = tf.Graph()
            tf.contrib.graph_editor.copy(sub_g_view, destgraph, reuse_dst_scope=True)
        return destgraph

    def change_inputs_data_format(self, inputs) :
        for i in range(len(self.inputs)) :
            if self.inputs[i] not in self.consts and isinstance(inputs[i], np.ndarray) and len(inputs[i].shape) == 4 :
                inputs[i] = np.transpose(inputs[i], [0,2,3,1])

    def forward_exec(self, inputs) :
        print(self.inputs, self.name_map)
        if self.mode == 'NCHW' :
            self.change_inputs_data_format(inputs)
        feed_dict = {self.name_map[self.inputs[i]+":0"] : inputs[i] for i in range(len(self.inputs))}
        with self.graph.as_default() :
            with tf.Session() as sess :
                output = sess.run(sess.graph.get_tensor_by_name(self.output+":0"), feed_dict)
                if self.mode == 'NCHW' and isinstance(output, np.ndarray) and len(output.shape) == 4 :
                    return np.transpose(output, [0,3,1,2])
                else :
                    return output

