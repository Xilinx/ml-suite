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

**Learn More:** [ML Suite Overview][]  
**Watch:** [Webinar on Xilinx FPGA Accelerated Inference][]

## Getting Started
1. Clone ML Suite    
  `git clone https://github.com/Xilinx/ml-suite.git` 
2. [Install Anaconda2][].  
  `# Ensure that you ran the fix_caffe_opencv_symlink.sh script`  
3. [Install git lfs](https://github.com/git-lfs/git-lfs/wiki/Installation)
4. Go into the ml-suite directory and pull down the models  
  `cd ml-suite; git lfs pull`
   
**TEMPORARY NOTE:**  
If you are evaluating on AWS, the binaries we have included support the latest Amazon Shell  
`DSA name:       xilinx_aws-vu9p-f1-04261818_dynamic_5_0`  
The Xilinx ml-suite AMI was bundled for an older shell  
For this reason, if you are starting your evaluation today, it is best to begin from the FPGA Developer AMI:  
If you are using the [AWS EC2 F1 FPGA DEVELOPER AMI](https://aws.amazon.com/marketplace/pp/B06VVYBLZZ) the following steps are necessary to setup the drivers:  
  
5. `git clone https://github.com/aws/aws-fpga.git`  
6. `cd aws-fpga`  
7. `source sdaccel_setup.sh`   
  
Remember that AWS requires users to run as root to control the FPGA, so the following is necessary to use Anaconda as root:

8. Become root `sudo su` 
9. Set Environment Variables Required by runtime `source <MLSUITE_ROOT>/overlaybins/setup.sh aws`  
10. Set User Environment Variables Required to run Anaconda `source ~centos/.bashrc`  
11. Activate the users Anaconda Virtual Environment`source activate ml-suite`  
  
You can avoid disk space problems on the FPGA DEVELOPER AMI by creating an instance with more than the default 70G of storage, or by resizing the /swapfile to something less than 35G. 

**Once your environment is set up, take a look at some of the command line tutorials and Jupyter Notebooks here:**
- [Tutorials][]


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
[ML Suite Overview]: docs/tutorials/ml-suite-overview.md
[Webinar on Xilinx FPGA Accelerated Inference]: https://event.on24.com/wcc/r/1625401/2D3B69878E21E0A3DA63B4CDB5531C23?partnerref=Mlsuite
