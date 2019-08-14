#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import sys, os

sys.path.append(os.environ['MLSUITE_ROOT'] + '/xfdnn/rt')
import xdnn, xdnn_io
import numpy as np
import json, copy

def generateLayerwiseJson(layername):
  #args = xdnn_io.processCommandLine()
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--layerindex', type=int, default=0, help='Index value for layer in json', required=True)
  argvt = parser.parse_args()
  args  = xdnn_io.make_dict_args(argvt)
  with open (args['netcfg'], 'r') as fp:
      data = json.load(fp)
  #print json.dumps(data, indent=2)
  # Get layers from json
  nodes = data['network']
  #print "Total layers (nodes): ", len(nodes)
  reachedNode = False
  for node in nodes:
      if node['active'] == 0:
          continue
      #print "Active: ", node['active'], " ", node['name']
      if reachedNode == False and node['name'] == layername:
          reachedNode = True
      elif reachedNode and node['name'] != layername:
          node['active'] = 0

  fname = str(layername) + str('.json')
  fjson = fname.replace('/', '_')
  with open(fjson, 'w') as wfp:
      json.dump(data, wfp, indent=2, sort_keys=True)
  return fjson 

def networkForward(netcfg, layername):

  #args = xdnn_io.processCommandLine()
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--layerindex', type=int, default=0, help='Index value for layer in json', required=True)
  argvt = parser.parse_args()
  args  = xdnn_io.make_dict_args(argvt)
  
  args['netcfg'] = netcfg
  # Hardcode these parameters, so we only have to look at performance of 1 PE
  args["batch_sz"] = 1
  args["PE"] = 0

  #print "{:-^100}".format(' Before: createHandle ')
  ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0")
  #print "{:-^100}".format(' After: createHandle ')
  if ret != 0:
      sys.exit(1)

  fpgaRT = xdnn.XDNNFPGAOp(handles, args)
  #print "{:-^100}".format('1')
  fpgaOutput = fpgaRT.getOutputs()
  #print "{:-^100}".format('2')
  fpgaInput = fpgaRT.getInputs()
  #print "{:-^100}".format('3')

  img_paths = xdnn_io.getFilePaths(args['images'])
  inShape = (args['batch_sz'],) +  tuple ( tuple (fpgaRT.getInputDescriptors().values() )[0][1:] )

  firstInput = list(fpgaInput.values())[0]
  firstOutput = list (fpgaOutput.values())[0] 


  for i in xrange(0, len(img_paths), args['batch_sz']):
    pl = []
    for j, p in enumerate(img_paths[i:i + args['batch_sz']]):
        firstInput[0, ...], _ = xdnn_io.loadImageBlobFromFile(img_paths[0], args['img_raw_scale'], args['img_mean'], args['img_input_scale'], inShape[2], inShape[3])
    pl.append(p)

    with open(args['netcfg']) as fp:
      data = json.load(fp)
      #print json.dumps(data, indent=2)

      # Strip nodes that don't run in hardware
      nodes = data['network']
      nodes = [x for x in nodes if x['xdnn_kv']]

      nLayers = len(nodes)

      # How many iterations to run, and average across
      iterations = 1

      # Initialize empty list to hold accumulated runtime
      t1 = []
      for k in range(iterations):
        t1.append(0.0)

      # Run N iterations of network permutations
      for l in range(iterations):
        fpgaRT.execute(fpgaInput, fpgaOutput)
        t1[l] += (fpgaRT.get_exec_time())

      #for node in nodes:
      #  print node['name']

      # Average it
      avetime = sum(t1)/iterations
      #print "{:<25} = {:<25}".format(layername, avetime)

  return avetime
  xdnn.closeHandle()
  del fpgaRT
  del fpgaInput
  del fpgaOutput
  del ret

def getCurrentLayerByIndex(index = 0):
  #args = xdnn_io.processCommandLine()
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--layerindex', type=int, default=0, help='Index value for layer in json', required=True)
  argvt = parser.parse_args()
  args = xdnn_io.make_dict_args(argvt)
  if 'layerindex' in args:
      index = args['layerindex']
  with open(args['netcfg']) as fp:
    data = json.load(fp)
    # Strip nodes that don't run in hardware
    nodes = data['network']
    nodes = [x for x in nodes if x['xdnn_kv'] and x['active'] == 1]
    # Get layername
    if index >= len(nodes):
        return None, None
    if nodes[index]['xdnn_kv']['slice'] == "0":
        return nodes[index]['name'], "DBL"

    
    return nodes[index]['name'], nodes[index]['xdnn_kv']['XNOp']


if __name__ == '__main__':
  parser = xdnn_io.default_parser_args()
  parser.add_argument('--layerindex', type=int, default=0, help='Index value for layer in json', required=True)
  argvt = parser.parse_args()
  args = xdnn_io.make_dict_args(argvt)
  #print json.dumps(args, indent=2)
  # Get layer name
  layername, opname = getCurrentLayerByIndex()
  if layername is None and opname is None:
    print "All = Done"
    sys.exit(0)
  if opname is not None and layername is None:
    print "DataMovementLayer = 0"
    sys.exit(0)
  if opname == "DBL":
    layername = layername + "-DBL"
    print layername,"= 0"
    sys.exit(0)
  # print "\n{:-^100}".format(layername)
  # Generate compiler JSON till this layer
  jsonname  = generateLayerwiseJson(layername)
  # print "\n{:-^100}".format(jsonname)
  # Get the latency of the network till this layer
  latency   = networkForward(jsonname, layername)
  print "{} = {}".format(layername, latency)
