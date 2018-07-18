# ML Suite Overview

The Xilinx ML Suite provides users the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, MxNet and Tensorflow as well as Python and RESTful APIs.

![](img/stack.png)

The ML Suite has three basic parts:
1. **xDNN IP** - High Performance general CNN processing engine.
2. **xfDNN Middleware** - Software Library and Tools to Interface with ML Frameworks and optimize them for Real-time Inference
3. **ML Framework and Open Source Support**  - Support for high level ML Frameworks and other open source projects

### xDNN IP
Xilinx xDNN IP cores are high performance general CNN processing engines. This means they can accept a wide range of CNN networks and models. xDNN has been optimized for different performance metrics. Today, there are two configurations available.

| DSP Array Configuration | Total Image Memory per PE | Total DSPs in Array | 16 bit GOP/s @500MHz | 8 bit GOP/s @500MHz |
|-------------------------|---------------------------|---------------------|----------------------|---------------------|
| 28x32                   | 4 MB                      | 896                 | 896                  | 1792                |
| 56x32                   | 6 MB                      | 1792                | 1792                 | 3584                |

The **28x32** configuration, also referred to as **medium**, is optimized for higher throughput. These kernels can be implemented on Xilinx FPGA fabric in different configurations. The **56x32** kernel is optimized for larger models, but also delivers lower latency.

Each xDNN IP Kernel supports the following Layers:

<p align="center">
  <img width="674" height="466" src="img/xdnnv2-support.png">
</p>

Below is an example of this, with an **overlay** of 4 **medium** kernels.

![](img/xdnn-overlay.png)

An **overlay** is a FPGA binary with multiple xDNN IP kernels along with necessary connectivity for on board DDR channels and communication with other system level components like PCIe. xDNN kernels can be combined with other accelerators, such as Video transcoding blocks, or custom IP blocks for custom overlays. For all the stanard overlays available in the ML Suite today, please take a look at the [Overlay Selector Guide][]

### xfDNN Middleware
xfDNN middleware is a high performance software library with a well-defined API which acts as a  bridge between deep learning frameworks such as Caffe, MxNet, Tensorflow and xDNN IP running on an FPGA. xfDNN software is currently the only available method of programming and using xDNN IP and assumes system running SDAccel reconfigurable acceleration stack compliant system.

xfDNN not provides simple python interfaces to connect to high level ML frameworks, but also provides tools for network optimization, by fusing layers, optimizing memory dependencies in the network and pre-scheduling the entire network removing CPU host control bottlenecks.

![](img/xfdnn-optimization.png)

Once these optimizations are completed per layer, the entire network is optimized for deployment in a "One-Shot" execution flow.

<p align="center">
  <img width="412" height="551" src="img/xfdnn-oneshot.png">
</p>

xfDNN Quantizer allows for quick high precision calibration to lower precision deployments to INT8 and INT16. These Python tools are simple to use.

For more information on this, please take a look at the xfDNN tool notebooks [here][].

### ML Framework and Open Source Support

The ML Suite supports the following Frameworks:
- [Caffe](https://caffe.berkeleyvision.org/)
- [Tensorflow](https://www.tensorflow.org/api_docs/)
- [Keras](https://keras.io/)
- [MXNet](https://mxnet.incubator.apache.org/api/python/index.html)
- [Darknet*](https://pjreddie.com/darknet/)  
    - Note: Darknet support is achieved by automatically converting to Caffe

With xfDNN you can connect to other Open Source frameworks and software easily with our Python APIs.
For more information on this, please take a look at the xfDNN tool notebooks [here][].

[Overlay Selector Guide]: ../../overlays/README.MD
[here]: ../../docs/tutorials/README.md
