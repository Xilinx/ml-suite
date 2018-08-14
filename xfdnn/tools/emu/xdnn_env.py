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

import xdnn
import json
import os, sys

class xdnn_env(object):
  def __init__(self):
    import xdnn
    self._xdnnParams = {}
    self._xdnnParams['lib_path'] = os.environ["LIBXDNN_PATH"]
    self._xdnnParams['api'] \
      = xdnn.XDNNManager(self._xdnnParams['lib_path'])
    self._xdnnParams['quantDB'] = None
    self._xdnnParams['scaleA'] = 10000
    self._xdnnParams['scaleB'] = 30
    self._xdnnParams['useGlobalScale'] = False

    if "XDNN_QUANTIZE_CFGFILE" in os.environ:
      quantFile = os.environ["XDNN_QUANTIZE_CFGFILE"]
      self._xdnnParams['quantize_json'] = quantFile
      with open(quantFile) as data:
        obj = json.load(data)
        # make a map of { layerName -> data }
        self._xdnnParams['quantDB'] = {}
        for l in obj['network']:
          self._xdnnParams['quantDB'][l['name']] = l
    else:
      self._xdnnParams['useGlobalScale'] = True

  def init_fpga(self, xclbin, isxdnnv3=False):
    self._xdnnParams['xclbin'] = xclbin
    self._xdnnParams['isXdnnv3']=isxdnnv3
    ret = xdnn.createHandle(
      self._xdnnParams['xclbin'], 
      "kernelSxdnn_0", 
      self._xdnnParams['lib_path'])

    if ret != 0:
      raise RuntimeError("Could not init FPGA %s %s" \
        % (xclbin, self._xdnnParams['lib_path']))
      sys.exit(1)

  def get_params(self): 
    return self._xdnnParams
