#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import xdnn
import json
import os, sys

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
                    self._xdnnParams['quantDB'][l['name']] = l
        else:
            self._xdnnParams['useGlobalScale'] = True

    def get_params(self): 
        return self._xdnnParams


class xdnn_fpga_env(xdnn_env):

    def __init__(self, xclbin, isxdnnv3=False):
        xdnn_env.__init__(self)
        self._xdnnParams['xclbin'] = xclbin
        self._xdnnParams['isXdnnv3'] = isxdnnv3
        self._xdnnParams['compiler_file'] = os.environ["XDNN_COMPILER_FILE"]
        ret = xdnn.createHandle(self._xdnnParams['xclbin'],
                                                        "kernelSxdnn_0",
                                                        self._xdnnParams['lib_path'])

        if ret != 0:
            raise RuntimeError("Could not init FPGA %s %s" % (xclbin, self._xdnnParams['lib_path']))
            sys.exit(1)

