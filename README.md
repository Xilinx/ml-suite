# Xilinx ML Suite

The Xilinx ML Suite enables developers to optimize and deploy accelerated ML inference.  It provides support for many common machine learning frameworks such as Caffe, MxNet and Tensorflow as well as Python and RESTful APIs.

The ML Suite includes:

- xfDNN compiler/optimizer – auto-layer fusing, memory optimization, and framework integration
- xfDNN quantizer – improves performance with auto model-precision INT8 calibration
- Platforms – deployable on-premise or through cloud services

### Getting Started
The ML Suite requires the installation of the Anaconda2 Virtual Environment. Here is a tutorial to [install Anaconda2][].

Once Anaconda2 is installed, create the virtual environment for ML Suite, by following [this guide][]

Once your environment is set up, take a look at some of the tutorials here:
- [Tutorials][]

### Minimum System Requirements
- OS: Ubuntu 16.04.2 LTS, CentOS
- CPU: 4 Cores (Intel/AMD)
- Memory: 8GB
- VCU1525 DSA Vesion: 5.0


### Supported Platforms
Cloud Services
 - [Amazon AWS EC2 F1][]

 On Premise Platforms
 - [Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit][]
    - Note: The `xilinx_vcu1525_dynamic_5_0` DSA is required to be installed. Installation information can be found on page 118 of [UG1023][]


### Release Notes
 - [1.0][]

### Questions and Support

- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]
- [SDAccel Forums][]


[install Anaconda2]: docs/tutorials/anaconda.md
[this guide]: docs/tutorials/start-anaconda.md
[Amazon AWS EC2 F1]: https://aws.amazon.com/marketplace/pp/B077FM2JNS
[Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit]: https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[Tutorials]: docs/tutorials/README.md
[1.0]: docs/release-notes/1.0.md
[UG1023]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_4/ug1023-sdaccel-user-guide.pdf
