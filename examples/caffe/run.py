from __future__ import print_function

import os,sys,argparse

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
from decent import CaffeFrontend as xfdnnQuantizer
from xfdnn_subgraph import CaffeCutter as xfdnnCutter

import numpy as np
import caffe

MLSUITE_ROOT = os.getenv("MLSUITE_ROOT","../../")
MLSUITE_PLATFORM = os.getenv("MLSUITE_PLATFORM","1525")

# Generate scaling parameters for fixed point conversion
def Quantize(prototxt,caffemodel,test_iter=1,calib_iter=1):
  quantizer = xfdnnQuantizer(
    model=prototxt,
    weights=caffemodel,
    test_iter=test_iter,
    calib_iter=calib_iter,
    auto_test=True,
  )
  quantizer.quantize()

# Standard compiler arguments for XDNNv3
def Getopts():
  return {
     "bytesperpixels":1,
     "dsp":96,
     "memory":9,
     "ddr":"256",
     "cpulayermustgo":True,
     "mixmemorystrategy":True,
     "pipelineconvmaxpool":True,
     "usedeephi":True,
  }

# Generate hardware instructions for runtime -> compiler.json
def Compile(prototxt="quantize_results/deploy.prototxt",\
            caffemodel="quantize_results/deploy.caffemodel",\
            quantize_info="quantize_results/quantize_info.txt"):
  compiler = xfdnnCompiler(
    networkfile=prototxt,
    weights=caffemodel,
    quant_cfgfile=quantize_info,
    generatefile="work/compiler",
    quantz="work/quantizer",
    **Getopts(),
  )
  compiler.compile()

# Generate a new prototxt with custom python layer in place of FPGA subgraph
def Cut(prototxt):
  cutter = xfdnnCutter(
    inproto="quantize_results/deploy.prototxt",
    trainproto=prototxt,
    outproto="xfdnn_auto_cut_deploy.prototxt",
    outtrainproto="xfdnn_auto_cut_train_val.prototxt",
    cutAfter="data",
    xclbin=MLSUITE_ROOT+"/overlaybins/"+MLSUITE_PLATFORM+"/overlay_4.xclbin",
    netcfg="work/compiler.json",
    quantizecfg="work/quantizer.json",
    weights="work/deploy.caffemodel_data.h5",
    profile=True
  )
  cutter.cut()

# Use this routine to evaluate accuracy on the validation set
def Infer(prototxt,caffemodel,numBatches=1):
  net = caffe.Net(prototxt,caffemodel,caffe.TEST)
  ptxtShape = net.blobs["data"].data.shape
  print ("Running with shape of: ",ptxtShape)
  results_dict = {}
  accum = {}
  for i in xrange(1,numBatches+1): 
    out = net.forward()
    for k in out:
      if out[k].size != 1:
        continue
      if k not in accum:
        accum[k] = 0.0 
      accum[k] += out[k]
      result = (k, " -- This Batch: ",out[k]," Average: ",accum[k]/i," Batch#: ",i)
      print (*result)
      if k not in results_dict:
        results_dict[k] = []
      results_dict[k].append(result)
  return results_dict

# Use this routine to classify a single image
def Classify(prototxt,caffemodel,image,labels):
  classifier = caffe.Classifier(prototxt,caffemodel,
    image_dims=[256,256], mean=np.array([104,117,123]),
    raw_scale=255, channel_swap=[2,1,0])
  predictions = classifier.predict([caffe.io.load_image(image)]).flatten()
  labels = np.loadtxt(labels, str, delimiter='\t')
  top_k = predictions.argsort()[-1:-6:-1]
  for l,p in zip(labels[top_k],predictions[top_k]):
    print (l," : ",p)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--prototxt', default="", help='User must provide the train_val prototxt')
  parser.add_argument('--caffemodel', default="", help='User must provide the caffemodel')
  parser.add_argument('--numBatches', type=int, default=1, help='User must provide number of batches to run')
  parser.add_argument('--qtest_iter', type=int, default=1, help='User can provide the number of iterations to test the quantization')
  parser.add_argument('--qcalib_iter', type=int, default=1, help='User can provide the number of iterations to run the quantization')
  parser.add_argument('--prepare', action="store_true", help='In prepare mode, model preperation will be perfomred = Quantize + Compile')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be ran on the FPGA, and the validation set examined')
  parser.add_argument('--image', default=None, help='User can provide an image to run')
  args = vars(parser.parse_args())
 
  if args["prepare"]: 
    Quantize(args["prototxt"],args["caffemodel"],args["qtest_iter"],args["qcalib_iter"])
    Compile()
    Cut(args["prototxt"])

  # Both validate, and image require that the user has previously called prepare.

  if args["validate"]: 
    Infer("xfdnn_auto_cut_train_val.prototxt",args["caffemodel"],args["numBatches"])

  if args["image"]:
    Classify("xfdnn_auto_cut_deploy.prototxt",args["caffemodel"],args["image"],"../deployment_modes/synset_words.txt")
  
