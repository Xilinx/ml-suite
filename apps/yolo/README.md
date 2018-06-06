# YOLOv2 Object Detection Tutorial

## Introduction
You only look once (YOLO) is a state-of-the-art, real-time object detection algorithm.  
The algorithm was published by Redmon et al. in 2016 via the following publications:
[YOLOv1](https://arxiv.org/abs/1506.02640),
[YOLOv2](https://arxiv.org/abs/1612.08242).  
The same author has already released YOLOv3, and some experimental tiny YOLO networks. We focus on YOLOv2.
This application requires more than just simple classification. The task here is to detect the presence of objects, and localize them within a frame. 
Please refer to the papers for full algorithm details, and/or watch [this.](https://www.youtube.com/watch?v=9s_FpMpdYW8). 
In this tutorial, the network was trained on the 80 class [COCO dataset.](http://cocodataset.org/#home)

## Background
The authors of the YOLO papers used their own programming framework called "Darknet" for research, and development. The framework is written in C, and was [open sourced.](https://github.com/pjreddie/darknet) Additionally, they host documentation, and pretrained weights [here.](https://pjreddie.com/darknet/yolov2/) Currently, the Darknet framework is not supported by Xilinx's ML Suite. Additionally, there are some aspects of the YOLOv2 network that are not supported by the Hardware Accelerator, such as the leaky ReLU activation function. For these reasons the network was modified, retrained, and converted to caffe. In this tutorial we will run the network accelerated on an FPGA using 16b quantized weights and a hardware kernel implementing a 56x32 systolic array with 5MB of image RAM. All convolutions/pools are accelerated on the FPGA fabric, while the final sigmoid, softmax, and non-max suppression functions are executed on the CPU. Converting from Darknet to Caffe will be discussed in future documentation.

### Network Modifications
* Leaky ReLU replaced by ReLU
* "reorg" layer a.k.a. "space_to_depth" layer replaced by MAX POOL
 
## Running the Application

Xilinx has provided a demo application showing how YOLOv2 can be ran "end to end", meaning we will run all of the required offline, and online software to get some example results.   
[yolo.py](./yolo.py) is the top level python module where you will see how the compiler, quantizer, and xyolo module are invoked.   
[configs.py](./configs.py) is a configuration file used to modify the desired run configuration. We are supporting 608x608/224x224 images, 16b/8b quantization.  
[xyolo.py](./xyolo.py) is a python class meant to be reusable, but it also demonstrates how to use the PYXDNN in a custom application. It provides a YOLO detect method.  
[run.sh](./run.sh) is a bash script used to run some enviornment setup, and launch the demo app  

 To run:
 1. Connect to F1 or Local Hardware
 
 2. Download the xilinx trained models from Xilinx.com, save as models at the root of this repo 
 
 3. `cd MLsuite/apps/yolo`

 4. `./run.sh <DEVICE> e2e`

 For example:  
 `./run.sh 1525 e2e`
 `./run.sh aws e2e`

Refer to the Using Anaconda on AWS instructions located [here][]. 

 Upon success, you will see several bounding box predictions printed for the images in the calibration directory

Note: After the first initial run, it is possible to run the demo with `python yolo.py` the run.sh script is setting up some key env variables, and building the non-max suppression binary. However, that only needs to be done once, in a shell

[here]: ../../docs/tutorials/start-anaconda.md
