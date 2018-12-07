#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import json
import tensorflow as tf
import numpy as np
from xfdnn_compiler_tensorflow import TFFrontend
#from xfdnn.tools.compile.frontends.frontend_caffe  import CaffeFrontend
from tensorflow.python.platform import gfile
import xdnn_opt

class xdnnRT:
    def __init__(self, compiler, rtargs):
        #print ("compiler args", cargs)
        self._inputs = self.list_inputs_of_graph()
        pydotGraph, schedule, self._out, ssize, compilerJson \
          = compiler.compile()
        with open('xdlfCompiler.json', 'w') as outfile:
             json.dump(compilerJson, outfile, sort_keys = True, indent = 4,
                            ensure_ascii = False)
#        print ("compiled pydot graph", pydotGraph)
#        print ("compiled schedule", schedule)

        opt = None
        if rtargs.device == "CPU":
            opt = xdnn_opt.CPUTransform( self._inputs, pydotGraph, schedule, rtargs)
        elif rtargs.device == "HWEmu":
            opt = xdnn_opt.HWEmuTransform( self._inputs, pydotGraph, schedule, rtargs)
        elif rtargs.device == "FPGA":
            if rtargs.xclbin:
                opt = xdnn_opt.FPGATransform( self._inputs, pydotGraph, schedule, compilerJson, rtargs)
            else:
                raise AttributeError("Must specify path to xclbin when device = FPGA")
        else:
            raise AttributeError("Unsupported device type", rtargs.device)
        #variables hold the inputs/consts of graph
        self._variables = opt.variables
        self._layers = opt.getLayers()
        for l in self._layers:
            l.setup()

    def list_inputs_of_graph(self):
        pass

    def preprocess(self,inputs):
        pass

    def batch_classify(self, img_list, batch, preprocess) :
        bctr = 0
        ictr = 0
        pred = None
        prepdata = {}
        prep = self._inputs[0]
        #print(len(img_list))
        ctr = 0
        pred = []
        while ctr < len(img_list) :
            ctrmax = min(ctr+batch, len(img_list))
            pred.append(self.feed_forward(img_list[ctr:ctrmax], preprocess = preprocess))
            ctr = ctrmax
        if len(pred) == 0 : return []
        elif len(pred) == 1 :
            return pred[0]
        return np.concatenate(pred)

    def feed_forward(self, inputs, out=None, preprocess=None):
        if not out:
            out = self._out[0]
        if not preprocess:
            preprocess = self.preprocess

        # Add network input to variables list
        self._variables[self._inputs[0]] = preprocess(inputs)

        for layer in self._layers:
            #print "CDBG :", layer.output, type(layer)
            layer_inputs = [self._variables[inp] for inp in layer.inputs]
            self._variables[layer.output] = layer.forward_exec( layer_inputs )
            #print self._variables[layer.output].shape

        return self._variables[out]

class TFxdnnRT(xdnnRT):
    def __init__ ( self, cargs):
        self._tfGraph = tf.GraphDef()
        with gfile.FastGFile(cargs.networkfile, 'rb') as f:
            self._tfGraph.ParseFromString(f.read())

        compiler = TFFrontend(cargs)

        xdnnRT.__init__(self, compiler, cargs)


    def list_inputs_of_graph(self) :
        res = []
        for node in self._tfGraph.node :
            if node.op == 'Placeholder' :
                res.append(node.name)
        return res

    def preprocess(self, inputs):
        if type(inputs) is not np.ndarray:
            inputs = np.transpose(self.read_tensor_from_image_file(inputs), [0,3,1,2])  # assuming that there is only one input
        return inputs

    def read_tensor_from_image_file(self, file_name,
                                    input_height=299,
                                    input_width=299,
                                    input_mean=0,
                                    input_std=255):
        input_name = "file_reader"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
            image_reader = tf.image.decode_png(file_reader, channels=3, name="png_reader")
        elif file_name.endswith(".gif"):
            image_reader = tf.squeeze(
            tf.image.decode_gif(file_reader, name="gif_reader"))
        elif file_name.endswith(".bmp"):
            image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
        else:
            image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
        float_caster = tf.cast(image_reader, tf.float32)
        dims_expander = tf.expand_dims(float_caster, 0)
        resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
        normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
        with tf.Session() as sess :
            result = sess.run(normalized)
        return result
