#!/bin/bash

export XDNN_QUANTIZE_CFGFILE=sf_quantize1.json
. ../../../../../overlaybins/setup.sh 1525
#. ../../../../../overlaybins/1525/setupv3.sh

#python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --image=cropped_panda.jpg #runs satyaflow

#python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --doHwEmu=True --image=cropped_panda.jpg #runs satyaflow+hwemuv2

python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --FPGA=True --custom=True --image=cropped_panda.jpg --xdnnv3=False --singleStep=True #runs satyaflow+fpga+singleStep 

#python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --FPGA=True --custom=True --image=cropped_panda.jpg --xdnnv3=False --singleStep=False #runs satyaflow+fpga+bunch 

#python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --FPGA=True --custom=True --image=cropped_panda.jpg --xdnnv3=True --singleStep=False #runs satyaflow+fpgav3+bunch 

#python run_tf.py

#python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --doHwEmu=True --image=cropped_panda.jpg --xdnnv3=True --bunch=True #runs satyaflow+hwemuv3+bunch

#python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt

# Resnet
#export XDNN_QUANTIZE_CFGFILE=resnet50_16b.json
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=False --image=cropped_panda.jpg #runs satyaflow
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --image=cropped_panda.jpg #runs satyaflow
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --doHwEmu=True --image=cropped_panda.jpg #runs satyaflow


# Single Layer Debug flow - fpga
#func()
#{
#  for i in $layernames; do
#      python run_tf.py --model=caffe_converted/googlenet_v1.pb --labels=caffe_converted/googlenet_v1_labels.txt --FPGA=True --custom=True --image=cropped_panda.jpg --xdnnv3=True --layerName $i #runs satyaflow+fpga 
#      shift
#  done
#
#}
#func


