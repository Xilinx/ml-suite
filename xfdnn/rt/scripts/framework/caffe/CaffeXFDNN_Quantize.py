#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import caffe

from xfdnn.rt.scripts.framework.base.quantize_controls import quantize_controls

class CaffeXFDNN_Quantize(caffe.Layer):
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
    outs = self._xdnn_env.quantize_inputs(inps, self.name)
    for i in range(len(outs)) :
      top[i].data[...] = outs[i]
    
    
  def backward(self, top, propagate_down, bottom):
    raise Exception("Can't do backward propagation... yet")
