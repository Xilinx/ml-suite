#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import cv2
import copy
import os.path
import math
import sys
import timeit
import xdnn, xdnn_io
import numpy as np
import types
import threading
import urllib2
from flask import Flask, jsonify, request

class XDNNEngine:
  def __init__(self, maxNumStreams):
    self._maxNumStreams = maxNumStreams 
    self._streamsAvailable = []
    self._streamInputs = []
    self._streamOutputs = []

    self._config = xdnn_io.processCommandLine()
    ret, handles = xdnn.createHandle(self._config['xclbin'])
    if ret != 0:
      sys.exit(1)

    self._fpgaRT = xdnn.XDNNFPGAOp(handles, self._config)
    self._fcWeight, self._fcBias = xdnn_io.loadFCWeightsBias(self._config)
    self._labels = xdnn_io.get_labels(self._config['labels'])

    for i in range(maxNumStreams):
      self._streamsAvailable.append(i)
      self._streamInputs.append(None)
      self._streamOutputs.append(None)

  def get_free_stream(self):
    return self._streamsAvailable.pop(0) if self._streamsAvailable else None

  def return_stream(self, streamId):
    self._streamsAvailable.append(streamId)

  def exec_async(self, image, streamId):
    args = copy.deepcopy(self._config)

    # prepare image
    image, _ = xdnn_io.loadImageBlobFromFile(image, 
      args['img_raw_scale'], args['img_mean'], 
      args['img_input_scale'], args['in_shape'][2], args['in_shape'][1])

    # initialize I/O buffers
    self._streamInputs[streamId] = np.empty(image.shape, 
      dtype=np.float32, order='C_CONTIGUOUS')
    np.copyto(self._streamInputs[streamId], image)
    if type(self._streamOutputs[streamId]) == type(None):
      self._streamOutputs[streamId] \
        = np.empty((1, args['fpgaoutsz'],), dtype=np.float32, order='C_CONTIGUOUS')

    # DO EET
    self._fpgaRT.exec_async(self._streamInputs[streamId], 
      self._streamOutputs[streamId], streamId)

  def exec_post_fpga(self, image, streamId):
    args = copy.deepcopy(self._config)
    fpgaOutput = self._streamOutputs[streamId]
    batch_sz = 1
    fcOut = np.empty((batch_sz, args['outsz'],), dtype=np.float32, 
      order='C_CONTIGUOUS')
    xdnn.computeFC(self._fcWeight, self._fcBias, 
      self._streamOutputs[streamId], 
      batch_sz, args['outsz'], args['fpgaoutsz'], fcOut)
  
    softmaxOut = xdnn.computeSoftmax(fcOut)

    result = xdnn_io.getClassification(softmaxOut, [image], self._labels)
    result = result.strip().split("\n")
    top5 = [x for x in result if "-------" not in x]
    return top5

  def get_result(self, image, streamId):
    self._fpgaRT.get_result(streamId)
    output = self.exec_post_fpga(image, streamId)
    return output

app = Flask(__name__)

g_engine = None 
def initEngine():
  global g_engine
  if g_engine == None:
    g_engine = XDNNEngine(1)

@app.route('/predict', methods=['POST'])
def predict():
  initEngine()

  streamId = None
  while streamId == None:
    streamId = g_engine.get_free_stream()

  result = { }

  try:
    imageUrl = request.form.get('url')
    resp = urllib2.urlopen(imageUrl)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    g_engine.exec_async(image, streamId)
    output = g_engine.get_result(image, streamId)
    result['predictions'] = output
    result['status'] = 'success'
  finally:
    g_engine.return_stream(streamId)

  return jsonify(result)

if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)
