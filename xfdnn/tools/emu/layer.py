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

class layer(object) :
  def setOp(self, operator) :
    self.op = operator

  def setInput(self, inputs) :
    self.inputs = inputs

  def setType(self, typename) :
    self.type = typename

  def setOutput(self, output) :
    self.output = output

  def setShape(self, shape):
    self.shape = shape
  
  def setup(self):
     pass
 
  def forward_exec_dbg(self, inputs):
     print("\n")
     print(inputs.output + " input ===========================================")
     print(inputs[0].shape)
     print(inputs[0])
     print("\n")
     out = self.forward_exec(inputs)
     print("\n")
     print(layer.output + " output ===========================================")
     print(out.shape)
     print(out)
     print("\n")
     return out
 
  def forward_exec(self, inputs) :
    sub_g = tf.Graph()
    with sub_g.as_default() :
        op = self.op
        #print sess.graph == sub_g, 'Session Graph : ', sess.graph, 'sub graph : ', sub_g
        sub_g_inputs = []
        for inp in inputs :
            #print inp.name[:inp.name.find(':')] in variables, op.type
            sub_g_inputs.append(tf.constant(inp))
            #print sub_g_inputs[-1].graph == sub_g
        newop = None
        if op.type == 'Conv2D' :
            newop = tf.nn.conv2d(sub_g_inputs[0], sub_g_inputs[1], op.get_attr('strides'), op.get_attr('padding'))
        elif op.type == 'BatchNormWithGlobalNormalization' :
            newop = tf.nn.batch_norm_with_global_normalization(sub_g_inputs[0], sub_g_inputs[1],sub_g_inputs[2], sub_g_inputs[3], 
                                    sub_g_inputs[4], variance_epsilon = op.get_attr('variance_epsilon'), 
                                    scale_after_normalization = op.get_attr('scale_after_normalization'))
        elif op.type == 'CheckNumerics' :
            newop = tf.check_numerics(sub_g_inputs[0], op.get_attr('message'))
        elif op.type == 'Identity' :
            newop = tf.identity(sub_g_inputs[0])
        elif op.type == 'Relu' :
            newop = tf.nn.relu(sub_g_inputs[0])
        elif op.type == 'AvgPool' : # this is reducing the efficiency
            newop = tf.nn.avg_pool(sub_g_inputs[0], op.get_attr('ksize'), op.get_attr('strides'), op.get_attr('padding'))
        elif op.type == 'MaxPool' :
            newop = tf.nn.max_pool(sub_g_inputs[0], op.get_attr('ksize'),  op.get_attr('strides'), op.get_attr('padding'))
        elif op.type == 'Concat' :
            newop = tf.concat(sub_g_inputs[1:], sub_g_inputs[0])
        elif op.type == 'ConcatV2' :
            newop = tf.concat(sub_g_inputs[:-1], sub_g_inputs[-1])
        elif op.type == 'Reshape' :
            newop = tf.reshape(sub_g_inputs[0], sub_g_inputs[1])
        elif op.type == 'MatMul' :
            newop = tf.matmul(sub_g_inputs[0], sub_g_inputs[1])
        elif op.type == 'BiasAdd' :
            newop = tf.nn.bias_add(sub_g_inputs[0], sub_g_inputs[1])
        elif op.type == 'Softmax' :
            newop = tf.nn.softmax(sub_g_inputs[0])
        elif op.type == 'AddN' :
            newop = tf.add(sub_g_inputs[0], sub_g_inputs[1])
        else : 
            print((op.type, 'not supported yet...'))
        with tf.Session() as mini_g_sess :
            #print mini_g_sess.graph == sub_g, 'Session Graph : ', sess.graph, 'sub graph : ', sub_g
            return mini_g_sess.run(newop)   
