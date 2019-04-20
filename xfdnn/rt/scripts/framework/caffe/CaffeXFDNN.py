#!/usr/bin/env python

import caffe,json
import xdnn, xdnn_io
import time
# Our custom FPGA One-shot layer
class CaffeXFDNN(caffe.Layer):

  # Called once when the network is initialized
  def setup(self, bottom, top):
    self.param_dict = eval(self.param_str) # Get args from prototxt
    self._args = xdnn_io.make_dict_args(self.param_dict)
    self._numPE = self._args["batch_sz"] # Bryan hack to detremine number of PEs in FPGA
    # Establish FPGA Communication, Load bitstream
    ret, handles = xdnn.createHandle(self._args["xclbin"], "kernelSxdnn_0")
    if ret != 0:
      raise Exception("Failed to open FPGA handle.")
    
    self._args["scaleB"] = 1    
    self._args["PE"] = -1    
    # Instantiate runtime interface object
    self.fpgaRT = xdnn.XDNNFPGAOp(handles, self._args)
    self._indictnames = self._args["input_names"]
    self._outdictnames =  self._args["output_names"]
    self._parser = xdnn.CompilerJsonParser(self._args["netcfg"])

  # Called before every forward
  def reshape(self, bottom, top):
    bsz = bottom[0].num
    for i,n in enumerate( self._indictnames ):
      dim = self._parser.getInputs()[n]
      dim[0] = bsz
      #print ( "INPUT NAME: ", n, "SHAPE: ", dim)
      t = tuple(dim)
      bottom[i].reshape (*t)
    
    for i,n in enumerate( self._outdictnames ):
      dim = self._parser.getOutputs()[ n ]
      dim[0] = bsz
      #print ( "OUTPUT NAME: ", n, "SHAPE: ", dim)
      t = tuple ( dim )
      top[i].reshape(*t)

    if self._args["profile"]:
      top[len(top)-1].reshape(1) # ONEHACK - Last top will always be latency

  # Called for every batch
  def forward(self, bottom, top):
    bsz = bottom[0].num
    indict = {}
    outdict = {}
    for i,n in enumerate(self._indictnames):
      indict[ n ] = bottom[i].data
      
    for i,n in enumerate(self._outdictnames):
      outdict[ n ] = top[i].data
      
    #for i in range(100):
    #  print (" RUNNING instr 0 -> ", i)
    #  self.fpgaRT.set_start_idx(0)
    #  self.fpgaRT.set_stop_idx(i)
    #t = time.time()
    
    #print ( "INDICT: ", indict )
    #print ( "OUTDICT: ", outdict)
    numiter = 1
    for i in range ( numiter ):
      self.fpgaRT.execute(indict, outdict)
      # Consider enabling this with switch
      if self._args["profile"]:
        avg_exec_time = self.fpgaRT.get_exec_time()
        divisor = bsz/self._numPE 
        if divisor != 0:
          avg_exec_time /= divisor
        top[len(top)-1].data[...] = avg_exec_time # Last top will always be latency

    #print ( ( (time.time() - t) * 1000) / numiter, " ms per run" )  

  def backward(self, top, propagate_down, bottom):
    raise Exception("Can't do backward propagation... yet")

