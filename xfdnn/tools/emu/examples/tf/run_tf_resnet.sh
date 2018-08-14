#!/bin/bash

export XDNN_QUANTIZE_CFGFILE=sf_quantize.json
. ../../../../../overlaybins/setup.sh 1525

# Resnet
#export XDNN_QUANTIZE_CFGFILE=resnet50_16b.json
unset XDNN_QUANTIZE_CFGFILE # use global scale
export XDNN_SLOT_TIMEOUT=1000
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=False --image=cropped_panda.jpg # default TF
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --image=cropped_panda.jpg # satyaflow
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --doHwEmu=True --image=cropped_panda.jpg # runs satyaflow + emu
#python run_tf.py --model=caffe_converted/resnet_50.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --FPGA=True --image=cropped_panda.jpg  # satyaflow + FPGA 

python run_tf.py --model=/wrk/acceleration/users/aaronn/tf_retrain/example_code/retrained_graph.pb --labels=caffe_converted/googlenet_v1_labels.txt --custom=True --image=cropped_panda.jpg  # satyaflow + FPGA 
