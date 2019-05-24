<table style="width:100%">
<tr>
<th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Xilinx ML Suite v1.4</h2>
</th>
</table>
  

The Xilinx Machine Learning (ML) Suite provides users with the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, Tensorflow, and MXNet.  

<p align="left">
  <img width="700" height="350" src="docs/img/stack.png">
</p>

The ML Suite is composed of three basic parts:
1. **ML Framework and Open Source Support**  - Support for high level ML Frameworks and other open source projects.
2. **xfDNN Middleware** - Software Library and Tools to Interface with ML Frameworks and optimize networks for Real-time Inference.
3. **xDNN IP** - High Performance CNN processing engine.

**Learn More:** [ML Suite Overview][]  
**Watch:** [Webinar on Xilinx FPGA Accelerated Inference][]   
**Forum:** [ML Suite Forum][]

## [See What's New](docs/release-notes/1.x.md)
 - [Release Notes][]
 - Integration of Deephi DECENT Quantizer for Caffe
 - xfDNN Runtime API upgraded to support multi-output networks
 - XDNNv3 fully integrated for all platforms & models
 - Ease of use enhancements
    - Docker Images
    - Run on FPGA using Caffe's custom Python layer
    - HDF5 format used for network weights
 
## Getting Started
 - [Install XRT](docs/xrt.md) (Only necessary for On-Premise deployment)
 - [Start Docker Container](docs/container.md)
 - [Jupyter Notebook Tutorials](notebooks/README.md)
   - [Caffe Image Classification](notebooks/image_classification_caffe.ipynb)
   - [Caffe Object Detection w/ YOLOv2](notebooks/object_detection_yolov2.ipynb)
 - Command Line Examples
   - [Caffe ImageNet Benchmark Models](examples/caffe/README.md)
   - [Caffe VOC SSD Example](examples/caffe/ssd-detect/README.md)
   - [Deployment Mode Examples](examples/deployment_modes/README.md)
 - [In-Browser GoogLeNet Demo](apps/perpetual_demo/README.md)
 - [REST Server Example](examples/caffe/REST/README.md)
 - [Container Pipeline Example](docs/container_pipeline.md)
 
## References 
- [ML Suite Overview][]  
- [Performance Whitepaper][]
- [Accuracy Benchmarks](examples/caffe/Benchmark_README.md)

## [System Requirements](https://github.com/Xilinx/XRT/blob/master/src/runtime_src/doc/toc/system_requirements.rst)

## Questions and Support
- [FAQ][]
- [ML Suite Forum][]
- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]

[models]: docs/models.md
[Amazon AWS EC2 F1]: https://aws.amazon.com/marketplace/pp/B077FM2JNS
[Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit]: https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[Release Notes]: docs/release-notes/1.x.md
[UG1023]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_4/ug1023-sdaccel-user-guide.pdf
[FAQ]: docs/faq.md
[ML Suite Overview]: docs/ml-suite-overview.md
[Webinar on Xilinx FPGA Accelerated Inference]: https://event.on24.com/wcc/r/1625401/2D3B69878E21E0A3DA63B4CDB5531C23?partnerref=Mlsuite
[ML Suite Forum]: https://forums.xilinx.com/t5/Xilinx-ML-Suite/bd-p/ML 
[ML Suite Lounge]: https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html
[Models]: https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html#gettingStartedCloud
[whitepaper here]: https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf
[Performance Whitepaper]: https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf
