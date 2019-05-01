#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import sys
import xdnn, xdnn_io
import numpy as np

import matplotlib.pyplot as plt


def main():
  args= xdnn_io.processCommandLine()

  # Hardcode these parameters, so we only have to look at performance of 1 PE
  args["batch_sz"] = 1
  args["PE"] = 0

  ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0")
  if ret != 0:
    sys.exit(1)
  fpgaRT = xdnn.XDNNFPGAOp(handles, args)
  fcWeight, fcBias = xdnn_io.loadFCWeightsBias(args)
  img_paths = xdnn_io.getFilePaths(args['images'])
  fpgaOutput = np.empty ((args['batch_sz'], args['fpgaoutsz'],), dtype=np.float32, order='C')
  fcOutput = np.empty((args['batch_sz'], args['outsz'],), dtype=np.float32, order='C')
  inShape = tuple(args['in_shape'])
  batch_array = np.empty(((args['batch_sz'],) + inShape), dtype=np.float32, order='C')
  labels = xdnn_io.get_labels(args['labels'])
  if args['golden']:
    goldenMap = xdnn_io.getGoldenMap(args['golden'])
    top5Count = 0
    top1Count = 0

  for i in xrange(0, len(img_paths), args['batch_sz']):
    pl = []
    for j, p in enumerate(img_paths[i:i + args['batch_sz']]):
      batch_array[j, ...], _ = xdnn_io.loadImageBlobFromFile(p, args['img_raw_scale'], args['img_mean'], args['img_input_scale'], inShape[2], inShape[1])
      pl.append(p)

    with open(args['netcfg']) as fp:
      import json
      data = json.load(fp)

    # Strip nodes that don't run in hardware
    nodes = data['network']
    nodes = [x for x in nodes if x['xdnn_kv']]

    nLayers = len(nodes)

    for node in nodes:
      print node['name']

    # Initialize empty list to hold accumulated runtime
    t1 = []
    for k in range(nLayers):
      t1.append(0.0)

    # How many iterations to run, and average across
    iterations = 10

    hack = 1
    # For now the runtime inserts a download instruction at the start of a schedule
    # The download instruction brings an image to URAM
    # This instruction does not appear in the compiler schedule
    # To compensate for this extra instruction in the queue, we need to step an extra 1

    # Run N iterations of network permutations
    for l in range(iterations):
      for k in range(nLayers):
        fpgaRT.set_stop_idx(k+hack)
        fpgaRT.execute(batch_array, fpgaOutput)
        t1[k] += (fpgaRT.get_exec_time())

    # Average it
    t1 = list(np.array(t1)/iterations)

    # Get the delta "td" contributed by each layer
    t0 = [0.0] + t1[0:-1]
    td = np.array(t1) - np.array(t0)

    print "Sanity check the sum of deltas: ",np.sum(td)
    print args['netcfg']
    
    if 'overlay_4' in args['xclbin']:
      dsp_freq = 700
      dsps = 96*16
    elif 'overlay_3' in args['xclbin']:
      dsp_freq = 500
      dsps = 56*32
    elif 'overlay_2' in args['xclbin']:
      dsp_freq = 500
      dsps = 56*32
    else:
      dsp_freq = 500
      dsps = 28*32

    with open(args['netcfg']+'.csv','w') as fp:
      fp.write(args['netcfg'] + ',\n')
      fp.write("cmd#,name,op,ops,time,ops/sec,efficiency,inchan,insize_h,insize_w,kernel_h,kernel_w,outchan,outsize_h,outsize_w,\n")
      for k in range(len(td)):
        if 'XNMaxPoolPipelined' in nodes[k]['xdnn_kv']['XNOp'] or 'XNConv' in nodes[k]['xdnn_kv']['XNOp']:
          fp.write(str(k) + ',')
          fp.write(str(nodes[k]['name']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['XNOp']) + ',')
          fp.write(str(nodes[k]['ops']) + ',')
          fp.write(str(td[k]) + ',')
          fp.write(str(1000*int(nodes[k]['ops'])/td[k]) + ',')
          fp.write(str(int(nodes[k]['ops'])/(4*dsps*dsp_freq*1000*td[k])) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['inchan']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['insize_h']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['insize_w']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['kernel_h']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['kernel_w']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['outchan']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['outsize_h']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['outsize_w']) + ',')
        else:
          fp.write(str(k) + ',')
          fp.write(str(nodes[k]['name']) + ',')
          fp.write(str(nodes[k]['xdnn_kv']['XNOp']) + ',')
          fp.write(str(nodes[k]['ops']) + ',')
          fp.write(str(td[k]) + ',')
          fp.write(str(1000*int(nodes[k]['ops'])/td[k]) + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
          fp.write("NA" + ',')
        fp.write('\n') 
    
    plt.plot(td)
    plt.show()
  
  xdnn.closeHandle()

if __name__ == '__main__':
    main()

