![](docs/tutorials/img/xlnx-ml-suite.png)  
  

The Xilinx Machine Learning (ML) Suite provides users with the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, Tensorflow, and MXNet.  

![](docs/tutorials/img/stack.png)

The ML Suite is composed of three basic parts:
1. **xDNN IP** - High Performance general CNN processing engine.
2. **xfDNN Middleware** - Software Library and Tools to Interface with ML Frameworks and optimize them for Real-time Inference.
3. **ML Framework and Open Source Support**  - Support for high level ML Frameworks and other open source projects.

**Learn More:** [ML Suite Overview][]  
**Watch:** [Webinar on Xilinx FPGA Accelerated Inference][]   
**Forum:** [ML Suite Forum][]

## Getting Started
1. Clone ML Suite    
  `git clone https://github.com/Xilinx/ml-suite.git` 
2. [Install Anaconda2][].  
  `# Ensure that you ran the fix_caffe_opencv_symlink.sh script`  
3. [Download Models](https://www.xilinx.com/member/forms/download/ml-suite-eula-xef.html?filename=models.zip)
4. If you are not running on aws, you will need to download the hardware overlay's for your system:  
  - [alveo-u200](https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html#gettingStartedU200)  
  - [1525](https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html#gettingStartedVCU1525)  
  `# Place them at: ml-suite/overlaybins/`  
  `i.e.`  
  `ml-suite/overlaybins/alveo-u200`
   
**Once you are set up, take a look through some of the provided references:**
- [ML Suite Overview][]  
- [Jupyter Notebooks](notebooks/)
- [Compiler Python API](docs/tutorials/api-xfdnncompile.md)
- [Quantizer Python API](docs/tutorials/api-xfdnnquantize.md)
- [xfDNN Python API](docs/tutorials/api-xfdnnruntime.md)
- [Precompiled Examples](examples/classification/README.md)  

## Recommended System Requirements
- OS: Ubuntu 16.04.2 LTS, CentOS 7.4
- CPU: 4 Cores (Intel/AMD)
- Memory: 8 GB

## Supported Platforms
Cloud Services
 - [Amazon AWS EC2 F1][]
 - [Nimbix](https://www.nimbix.net/xilinx/)

 On Premise Platforms
  - [Alveo U200 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html#gettingStartedU200)
 - [Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit][]
    - Note: The `xilinx_vcu1525_dynamic_5_1` DSA is required to be installed. Installation information can be found on page 118 of [UG1023][]

## Release Notes
 - [1.0][]
 - [1.1][]
 - [1.2][]
## Questions and Support

- [FAQ][]
- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]
- [ML Suite Forum][]


[install Anaconda2]: docs/tutorials/anaconda.md
[models]: docs/tutorials/models.md
[Amazon AWS EC2 F1]: https://aws.amazon.com/marketplace/pp/B077FM2JNS
[Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit]: https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[Tutorials]: docs/tutorials/README.md
[1.0]: docs/release-notes/1.0.md
[1.1]: docs/release-notes/1.1.md
[1.2]: docs/release-notes/1.2.md
[UG1023]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_4/ug1023-sdaccel-user-guide.pdf
[FAQ]: docs/tutorials/faq.md
[ML Suite Overview]: docs/tutorials/ml-suite-overview.md
[Webinar on Xilinx FPGA Accelerated Inference]: https://event.on24.com/wcc/r/1625401/2D3B69878E21E0A3DA63B4CDB5531C23?partnerref=Mlsuite
[ML Suite Forum]: https://forums.xilinx.com/t5/Xilinx-ML-Suite/bd-p/ML 

