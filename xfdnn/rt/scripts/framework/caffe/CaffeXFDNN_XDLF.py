#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env python

import caffe
import time

from xfdnn.rt.xdnn_rt_compiled import compilerxdnnRT
from xfdnn.rt import xdnn, xdnn_io

# Our custom FPGA One-shot layer
class CaffeXFDNN_XDLF(caffe.Layer):

  # Called once when the network is initialized
  def setup(self, bottom, top):
    
    _args = eval(self.param_str)

    #args = xdnn_io.make_dict_args(param_dict)
    if 'save' in _args and len(_args['save']) > 0 :
      import os.path as osp
      import os
      if not osp.exists(_args['save']) : os.makedirs(_args['save'])
    self.rt = compilerxdnnRT(_args['netcfg'], _args['weights'], _args['device'], _args["input_names"], _args["output_names"],  _args['xdnnv3'], _args['save'])
    self._indictnames = _args["input_names"]
    self._outdictnames =  _args["output_names"]
    self._parser = xdnn.CompilerJsonParser(_args["netcfg"])

  # Called before every forward
  def reshape(self, bottom, top):
    bsz = bottom[0].num
    
    for i,n in enumerate( self._indictnames ):
      dim = self._parser.getInputs()[n]
      dim[0] = bsz
      #print ( "NAME: ", n, "SHAPE: ", dim)
      t = tuple(dim)
      bottom[i].reshape (*t)
    
    for i,n in enumerate( self._outdictnames ):
      dim = self._parser.getOutputs()[ n ]
      dim[0] = bsz
      #print ( "NAME: ", n, "SHAPE: ", dim)
      t = tuple ( dim )
      top[i].reshape(*t)

  # Called for every batch
  def forward(self, bottom, top):
    indict = {}
    outdict = {}
    inps = []
    for i,n in enumerate(self._indictnames):
      inps.append(bottom[i].data)
      
      
    #for i in range(100):
    #  print (" RUNNING instr 0 -> ", i)
    #  self.fpgaRT.set_start_idx(0)
    #  self.fpgaRT.set_stop_idx(i)
    #t = time.time()
    
    #print ( "INDICT: ", indict )
    #print ( "OUTDICT: ", outdict)
    #numiter = 1
    #for i in range ( numiter ):
    #  self.fpgaRT.execute(indict, outdict)

    self.rt.forward_exec(inps)

    for i,n in enumerate(self._outdictnames):
      top[i].data[:] = self.rt.variables[n]
      
    #print ( ( (time.time() - t) * 1000) / numiter, " ms per run" )  

  def backward(self, top, propagate_down, bottom):
    raise Exception("Can't do backward propagation... yet")

