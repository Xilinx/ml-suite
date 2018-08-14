
import tensorflow as tf
import numpy as np
from xfdnn_compiler_tensorflow import TFFrontend
#from xfdnn.tools.compile.frontends.frontend_caffe  import CaffeFrontend
from tensorflow.python.platform import gfile
from . import xdnn_opt

class xdnnRT:
    def __init__(self, compiler, rtargs):
        #print ("compiler args", cargs)
        self._inputs = self.list_inputs_of_graph()
        pydotGraph, schedule, self._out, _ = compiler.compile()
        
#         print ("compiled pydot graph", pydotGraph)
#         print ("compiled schedule", schedule)     

        opt = None
        if rtargs.device == "CPU":   
            opt = xdnn_opt.CPUTransform( self._inputs, pydotGraph, schedule)
        elif rtargs.device == "FPGA":
            if rtargs.xclbin:
                opt = xdnn_opt.FPGATransform( self._inputs, pydotGraph, schedule, rtargs.xclbin)
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
    
    def feed_forward(self, inputs, out=None):
        inputs = self.preprocess(inputs)
        for k, v in list(inputs.items()):
            self._variables[k] = v
            
        for layer in self._layers:
            layer_inputs = []
            layer_inputs = [self._variables[inp] for inp in layer.inputs]
            self._variables[layer.output] = layer.forward_exec( layer_inputs )
            
        if out is None:
            return self._variables[self._out]
        
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
            idict = {}
            idict[ self._inputs[0]] = np.transpose(self.read_tensor_from_image_file(inputs), [0,3,1,2])     # assuming that there is only one input
            inputs = idict
        return inputs    

    def read_tensor_from_image_file(self, file_name,
                                    input_height=299,
                                    input_width=299,
                                    input_mean=0,
                                    input_std=255):
        input_name = "file_reader"
        file_reader = tf.read_file(file_name, input_name)
        if file_name.endswith(".png"):
          image_reader = tf.image.decode_png(
              file_reader, channels=3, name="png_reader")
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
     
