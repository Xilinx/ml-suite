#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import xdnn
import json
import os, sys

def _fixQuantJsonLayerNames(quantFile):
  obj = None
  with open(quantFile) as data:
      obj = json.load(data)

  for l in obj['network']:
    if "/Conv2D" not in l['name']:
      l['name'] = l['name']+"/Conv2D"

  with open(quantFile + "_fixed", "w") as outfile:
    json.dump(obj, outfile, sort_keys=True, indent=4, separators=(',', ': '))

class xdnn_env(object):
    def __init__(self):
        self._xdnnParams = {}
        self._xdnnParams['lib_path'] = os.environ["LIBXDNN_PATH"]
        self._xdnnParams['api'] = xdnn.XDNNManager(self._xdnnParams['lib_path'])
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
                  layerName = l['name']
                  self._xdnnParams['quantDB'][layerName] = l
        else:
            self._xdnnParams['useGlobalScale'] = True

    def get_params(self):
        return self._xdnnParams


class xdnn_fpga_env(xdnn_env):
    def __init__(self, xclbin, isxdnnv3=False):
        xdnn_env.__init__(self)
        self._xdnnParams['xclbin'] = xclbin

        if isxdnnv3 and 'v3' not in os.environ['LIBXDNN_PATH']:
          os.environ['LIBXDNN_PATH'] += '.v3'

        self._xdnnParams['isXdnnv3'] = isxdnnv3

        if "XDNN_COMPILER_FILE" in os.environ:
          self._xdnnParams['compiler_file'] = os.environ["XDNN_COMPILER_FILE"]
        (ret, handles) = xdnn.createHandle(self._xdnnParams['xclbin'])
        self._xdnnParams['handles'] = handles

        if ret != 0:
            raise RuntimeError("Could not init FPGA %s %s" % (xclbin, self._xdnnParams['lib_path']))
            sys.exit(1)
