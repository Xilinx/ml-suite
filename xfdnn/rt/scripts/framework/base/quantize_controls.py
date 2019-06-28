import os

import numpy as np
import collections

from xfdnn.rt.xdnn_env import xdnn_env

class quantize_controls(xdnn_env):
    def __init__(self, qcfg, xdnn_lib = None):
        if xdnn_lib == None or not os.path.exists(xdnn_lib) :
            super(quantize_controls, self).__init__(quant_cfgfile=qcfg)
        else :
            super(quantize_controls, self).__init__(quant_cfgfile=qcfg, lib_path=xdnn_lib)

    def quantize_inputs(self, inputs, name) :
        res = []
        for inp in inputs :
            blob = np.copy(inp)
            if self._xdnnParams['useGlobalScale']:
                res.append(blob * xdnnParams['scaleB'])
            else :
                blob = np.ascontiguousarray(inp, dtype=np.float32)
                qp = self._xdnnParams['quantDB'][name]
                self._xdnnParams['api'].quantizeTensor(qp['th_layer_in'], qp['bw_params'], blob)
                res.append(blob)
        return res

    def unquantize_outputs(self, outputs, name) :
        res = []
        for out in outputs :
            blob = np.copy(out)
            if self._xdnnParams['useGlobalScale']:
                res.append(blob / xdnnParams['scaleB'])
            else :
                blob = np.ascontiguousarray(blob, dtype=np.float32)
                qp = self._xdnnParams['quantDB'][name]
                self._xdnnParams['api'].unquantizeTensor(qp['th_layer_out'], qp['bw_layer_out'], blob)
                res.append(blob)

        return res

    def quantize_wts(self, weights, name):                                   
      if self._xdnnParams['useGlobalScale']:
        weights[...]= weights * self._xdnnParams['scaleA']                              
      else:                                                                       
        qp = self._xdnnParams['quantDB'][name]
        myWeights = np.ascontiguousarray(weights, dtype=np.float32)
        for thresh_out, weight in zip(qp['th_params'], myWeights):
          self._xdnnParams['api'].quantizeWeights(thresh_out, qp['bw_params'], weight)
        if qp['bw_layer_out'] == 16:
          weights = weights.astype(np.int16)
        elif qp['bw_layer_out'] == 8:
          weights = weights.astype(np.int8)

    def bitcorrection(self, inps, name):
      res = []
      qp = self._xdnnParams['quantDB'][name]
      for inp in inps :
        result = np.copy(inp)
        if qp['bw_layer_out'] == 8:
          pos_bitwidths = np.ceil(np.log2(np.where(result > 0, result, 1)))
          neg_bitwidths = np.ceil(np.log2(np.where(result < 0, -result, 1)))
          bitwidths = pos_bitwidths + neg_bitwidths
          if np.any(bitwidths>24):
            print(('WARNING: accumulation overflow detected: max_bitwidth '
                  '{}, max_pos {}, min_neg {}').format(bitwidths.max(),
                  result[bitwidths>24].max(), result[bitwidths>24].min()))
        if self._xdnnParams['useGlobalScale']:
          result = result / self._xdnnParams['scaleA']
        else:
          # transpose to group by "channel"
          myResult = np.ascontiguousarray(result, dtype=np.longlong)
          for ci in range(myResult.shape[0]):
            self._xdnnParams['api'].quantizev3InterLayer(qp['prescale_shift'][ci], qp['scale'][ci], qp['postscale_shift'][ci], qp['bw_params'], myResult[ci], 0)
          result = myResult
        res.append(result)
      return res

    def quantize_bias(self, bias, name, useGlobalScale = False):
      if self._xdnnParams['useGlobalScale']:
        bias[...] = bias * self._xdnnParams['scaleB']
      else:
        qp = self._xdnnParams['quantDB'][name]
        if isinstance(bias, collections.Iterable):
          f = lambda x: self._xdnnParams['api'].quantizeBias(\
            qp['th_layer_out'], qp['bw_params'], x)
          for x in np.nditer(bias, op_flags=['readwrite']):
            x[...] = f(x)
        else:
          bias = self._xdnnParams['api'].quantizeBias(\
            qp['th_layer_out'], qp['bw_params'], bias)
        #print "bias max", np.max(self.biases)
        if qp['bw_layer_out'] == 16:
          bias = bias.astype(np.int16)
        else:
          bias = bias.astype(np.int8)
        #print "biases",self.biases.shape,self.biases.dtype

