# Xilinx Caffe End-to-End Description

## Description of Individual Components (See Flow Diagram Below)

**Offline Components - Only Run Once; Not involved in benchmarking**
* Compiler
  * Inputs:
    * Darknet Weights (.weights)
    * Darknet Graph (.cfg)
  * Outputs
    * FPGA Instructions
    * Caffe Graph (.prototxt)
    * Caffe Weights (.caffemodel)
    * Weights Folder (.txt files)
* Quantizer
  * Inputs: 
    * Caffe Graph from Compiler (.prototxt)
    * Caffe Weights from Compiler (.caffemodel) 
  * Outputs:
    * Quantization Parameters (.json)

**Online Components - Run Iteratively; Used for benchmarking**
* PYXFDNN: Python overlay interface
  * Inputs: 
    * FPGA Instructions from Compiler
    * Quantization Parameters from Quantizer (.json)
    * Weights Folder from Compiler (.txt files) 
  * Backend: XFDNN (C++) --> SDACCEL Runtime --> Hardware (xclbin) 

![Complete Flow Diagram](img/caffe_flow.png)


