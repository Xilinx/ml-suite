
## Indtroduction 
You only look once (YOLO) is a state-of-the-art, real-time object detection algorithm.  
The algorithm was published by Redmon et al. in 2016 via the following publications:
[YOLOv1](https://arxiv.org/abs/1506.02640),
[YOLOv2](https://arxiv.org/abs/1612.08242).  
The same author has already released YOLOv3, and some experimental tiny YOLO networks. We focus on YOLOv2.  
This application requires more than just simple classification. The task here is to detect the presence of objects, and localize them within a frame.  
Please refer to the papers for full algorithm details, and/or watch [this.](https://www.youtube.com/watch?v=9s_FpMpdYW8) 
In this tutorial, the network was trained on the 80 class [COCO dataset.](http://cocodataset.org/#home)

## Background
The authors of the YOLO papers used their own programming framework called "Darknet" for research, and development. 
The framework is written in C, and was [open sourced.](https://github.com/pjreddie/darknet)
Additionally, they host documentation, and pretrained weights [here.](https://pjreddie.com/darknet/yolov2/)
Currently, the Darknet framework is not supported by Xilinx's ML Suite.
Additionally, there are some aspects of the YOLOv2 network that are not supported by the Hardware Accelerator, such as the leaky ReLU activation function. For these reasons the network was modified, retrained, and converted to caffe. In this tutorial we will run the network accelerated on an FPGA using 16b quantized weights and a hardware kernel implementing a 56x32 systolic array with 5MB of image RAM. All convolutions/pools are accelerated on the FPGA fabric, while the final sigmoid, softmax, and non-max suppression functions are executed on the CPU.  

## Network Manual Modifications (Implemented directly in Darknet)
To accommodate FPGA acceleration, two modifications are required for the YOLOv2 network, and then retraining must occur: 
* Leaky ReLU replaced by ReLU (The FPGA HW does not support Leaky ReLU)
* "reorg" layer a.k.a. "space_to_depth" layer replaced by MAX POOL (W/H Dimensionality is maintained)
  * The reorg layer does not appreciably improve accuracy, and incurrs costly data movement penalties.
  * Converting to max pool preserves the dimensionality required for downstream layers.
  ``` 
  # Converting below layer to maxpool to preserve downstream HxW relationships
  # [reorg]
  # stride=2
  [maxpool]
  size=2
  stride=2
  ```