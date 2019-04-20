from __future__ import print_function

import os,sys,argparse

from xfdnn.tools.compile.bin.xfdnn_compiler_caffe  import CaffeFrontend as xfdnnCompiler
from decent import CaffeFrontend as xfdnnQuantizer
from xfdnn_subgraph import CaffeCutter as xfdnnCutter
import caffe

MLSUITE_ROOT = os.getenv("MLSUITE_ROOT","../../")
MLSUITE_PLATFORM = os.getenv("MLSUITE_PLATFORM","1525")

def Quantize(prototxt,caffemodel,test_iter=1,calib_iter=1):
  quantizer = xfdnnQuantizer(
    model=prototxt,
    weights=caffemodel,
    test_iter=test_iter,
    calib_iter=calib_iter,
    auto_test=True,
  )
  quantizer.quantize()

def Getopts():
  return [\
     "--bytesperpixels","1", \
     "--cpulayermustgo", \
     "--mixmemorystrategy", \
     "--pipelineconvmaxpool", \
     "--usedeephi", \
     "--dsp","96", \
     "--memory","9", \
     "--ddr","256" \
  ]
           
def Compile(prototxt="quantize_results/deploy.prototxt",\
            caffemodel="quantize_results/deploy.caffemodel",\
            quantize_info="quantize_results/quantize_info.txt"):
  compiler = xfdnnCompiler(
    Getopts(),
    networkfile=prototxt,
    weights=caffemodel,
    quant_cfgfile=quantize_info,
    generatefile="work/compiler",
    quantz="work/quantizer"
  )
  compiler.compile()
  
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
  
def InferImage(prototxt,caffemodel,image,labels):
  import numpy as np
  import xdnn_io
  net = caffe.Net(prototxt,caffemodel,caffe.TEST)
  transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
  transformer.set_transpose('data', (2,0,1))
  transformer.set_mean('data', np.array([104,117,123]))
  transformer.set_raw_scale('data', 255)
  transformer.set_channel_swap('data', (2,1,0)) # if using RGB instead if BGR
  img=caffe.io.load_image(image)
  net.blobs['data'].data[...] = transformer.preprocess('data',img)
  ptxtShape = net.blobs["data"].data.shape
  print ("Running with shape of: ",ptxtShape)
  out = net.forward()
  for key in out:
    try:
      if out[key].shape[1] == 1000:
        softmax = out[key]
    except:
      pass
  Labels = xdnn_io.get_labels(labels)
  xdnn_io.printClassification(softmax,[image],Labels)

def Compare(cpu_train_val,fpga_train_val,caffemodel,numBatches=1,tolerance=10):
  cpuResults  = Infer(cpu_train_val,caffemodel,numBatches)
  fpgaResults = Infer(fpga_train_val,caffemodel,numBatches)
  for k in cpuResults:
    if k in fpgaResults:
      for i in range(len(cpuResults[k])):
        percent_difference = 100*abs(cpuResults[k][i][2]-fpgaResults[k][i][2])/cpuResults[k][i][2]
        print ("CPU: Iteration ",i,": ",cpuResults[k][i],"\nFPGA: Iteration ",i,": ",fpgaResults[k][i]," Percent Difference: ",percent_difference)
        if percent_difference > tolerance:
          raise ValueError("FPGA differs from cpu by more than given tolerance.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--train_val', default="", help='User must provide the train_val prototxt')
  parser.add_argument('--deploy', default="", help='User can provide the train_val prototxt')
  parser.add_argument('--caffemodel', default="", help='User must provide the caffemodel')
  parser.add_argument('--numBatches', type=int, default=1, help='User must provide number of batches to run')
  parser.add_argument('--qtest_iter', type=int, default=1, help='User can provide the number of iterations to test the quantization')
  parser.add_argument('--qcalib_iter', type=int, default=1, help='User can provide the number of iterations to run the quantization')
  parser.add_argument('--prepare', action="store_true", help='In prepare mode, model preperation will be perfomred = Quantize + Compile')
  parser.add_argument('--validate', action="store_true", help='If validation is enabled, the model will be ran on the FPGA, and the validation set examined')
  parser.add_argument('--compare', action="store_true", help='If validation is enabled, the model will be ran on the FPGA, and the validation set examined')
  parser.add_argument('--cpu', action="store_true", help='In cpu mode, we will just run the original CPU model')
  parser.add_argument('--image', default=None, help='User can provide an image to run')
  args = vars(parser.parse_args())
 
  if args["prepare"]: 
    Quantize(args["train_val"],args["caffemodel"],args["qtest_iter"],args["qcalib_iter"])
    Compile()
    Cut(args["train_val"])
 
  if args["validate"]: 
    if args["cpu"]: 
      Infer(args["train_val"],args["caffemodel"])
    else:
      Infer("xfdnn_auto_cut_train_val.prototxt",args["caffemodel"],args["numBatches"])

  if args["compare"]: 
    Compare(args["train_val"],"xfdnn_auto_cut_train_val.prototxt",args["caffemodel"])

  if args["image"]:
    if args["cpu"]: 
      InferImage(args["deploy"],args["caffemodel"],args["image"],"../classification/synset_words.txt")
      # We assume you want to run the original model, but you can run the optimized model with the below
      #InferImage("quantize_results/deploy.prototxt","quantize_results/deploy.caffemodel",args["image"],"../classification/synset_words.txt")
    else:
      InferImage("xfdnn_auto_cut_deploy.prototxt",args["caffemodel"],args["image"],"../classification/synset_words.txt")
  
