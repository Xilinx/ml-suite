from __future__ import print_function

import os,sys,argparse
import caffe
import io
import numpy as np
import xdnn_io

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
  return classifier

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--caffemodel', default="/opt/ml-suite/share/quantize_results/deploy.caffemodel", help='path to caffe model eg: /opt/ml-suite/share/quantize_results/deploy.caffemodel')
  parser.add_argument('--prototxt', default="/opt/ml-suite/share/xfdnn_auto_cut_deploy.prototxt", help='path to  prototxt file eg: /opt/ml-suite/share/xfdnn_auto_cut_deploy.prototxt')
  parser.add_argument('--synset_words', default="/opt/ml-suite/examples/deployment_modes/synset_words.txt", help='path to synset_words eg: /opt/ml-suite/examples/deployment_modes/synset_words.txt')
  parser.add_argument('--image', default="/opt/ml-suite/examples/deployment_modes/dog.jpg")
  args = vars(parser.parse_args())

  if args["caffemodel"]:
    model=args["caffemodel"]
  if args["prototxt"]:
    prototxt=args["prototxt"]
  if args["synset_words"]:
    synset_words=args["synset_words"]
  if args["image"]:
    image=args["image"]

  print("Loading FPGA with image and classify...")
  Classify(prototxt, model, image, synset_words) 
