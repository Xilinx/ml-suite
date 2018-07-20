<table style="width:100%">
<tr>
<th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Xilinx ML Suite</h2>
</th>
</table>

The Xilinx Machine Learning (ML) Suite provides users with the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, MxNet and Tensorflow as well as Python and RESTful APIs.

![](docs/tutorials/img/stack.png)

The ML Suite is composed of three basic parts:
1. **xDNN IP** - High Performance general CNN processing engine.
2. **xfDNN Middleware** - Software Library and Tools to Interface with ML Frameworks and optimize them for Real-time Inference.
3. **ML Framework and Open Source Support**  - Support for high level ML Frameworks and other open source projects.

**Get familiar with the [ML Suite Here][]**

## Getting Started
1. [Install Anaconda2][].
2. [Install git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
3. Clone this repo  
  `git clone https://github.com/Xilinx/ml-suite.git`  
  
If you are using the [AWS EC2 F1 FPGA DEVELOPER AMI](https://aws.amazon.com/marketplace/pp/B06VVYBLZZ) the following steps are necessary to setup the drivers:  
1. `git clone https://github.com/aws/aws-fpga.git`
2. `cd aws-fpga`
3. `source sdaccel_setup.sh`   
Once your environment is set up, take a look at some of the command line tutorials and Jupyter Notebooks here:
- [Tutorials][]

To run the tutorials you will need to download the trained [models]. 

## Minimum System Requirements
- OS: Ubuntu 16.04.2 LTS, CentOS
- CPU: 4 Cores (Intel/AMD)
- Memory: 8 GB

## Supported Platforms
Cloud Services
 - [Amazon AWS EC2 F1][]
 - [Nimbix](https://www.nimbix.net/xilinx/)

 On Premise Platforms
 - [Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit][]
    - Note: The `xilinx_vcu1525_dynamic_5_1` DSA is required to be installed. Installation information can be found on page 118 of [UG1023][]


## Release Notes
 - [1.0][]
 - [1.1][]

## Questions and Support

- [FAQ][]
- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]
- [SDAccel Forums][]


[install Anaconda2]: docs/tutorials/anaconda.md
[models]: docs/tutorials/models.md
[Amazon AWS EC2 F1]: https://aws.amazon.com/marketplace/pp/B077FM2JNS
[Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit]: https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[Tutorials]: docs/tutorials/README.md
[1.0]: docs/release-notes/1.0.md
[1.1]: docs/release-notes/1.1.md
[UG1023]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_4/ug1023-sdaccel-user-guide.pdf
[FAQ]: docs/tutorials/faq.md
[ML Suite here]: docs/tutorials/ml-suite-overview.md
