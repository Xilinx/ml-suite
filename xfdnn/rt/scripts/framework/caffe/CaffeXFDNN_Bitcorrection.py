import caffe,json
import xdnn, xdnn_io
from quantize_controls import quantize_controls
import time

class CaffeXFDNN_Bitcorrection(caffe.Layer):
  def setup(self, bottom, top):
    _args = eval(self.param_str) # Get args from prototxt
    self._xdnn_env = quantize_controls(qcfg=_args['quantizecfg'], xdnn_lib=_args['xdnn_lib'])
    params = self._xdnn_env.get_params()
    self.name = _args["name"]

  def reshape(self, bottom, top):
    for i in range(len(bottom)):
      dim = bottom[i].data.shape
      top[i].reshape(*dim)

  def forward(self, bottom, top):
    inps = [bottom[i].data for i in range(len(bottom))]
    tops = self._xdnn_env.bitcorrection(inps, self.name)
    for i in range(len(tops)) :
        top[i].data[...] = tops[i]
    
    
  def backward(self, top, propagate_down, bottom):
    raise Exception("Can't do backward propagation... yet")
