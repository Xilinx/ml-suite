# Xilinx AWS F1 xfDNN Test Drive

### Getting Started

1. Request Access to [Xilinx AWS F1 xfDNN Test Drive here][]
2. Launch Test Drive Instance and connect to your instance
	[Launching Instructions][]
3. Try Tutorials

### Tutorials
- [Image Classification GoogLeNet v1 Demo][]
- [DeepDetect REST Tutorial][]
- [DeepDetect Webcam Tutorial][]
- Network Compiler / Anayzer
	- Detailed Readme in the AMI:/`xfdnn_testdrive/xfdnn_compiler/README.rst`
	- Refer to `xfdnn_testdrive/xfdnn_compiler/docs/` for more information



### Latest AMI Version

- xfDNN-Preview-0.3b
	- Features:
		- 8/16-bit GoogLeNet v1
		- 8/16-bit ResNet50
		- Caffe
		- Image Classification
		- DeepDetect REST Demo
		- DeepDetect Webcam Demo
		- Caffe Network Analyzer
	- Notes:
		- Release 10/23/17
		- Known issues
			- Performance Limitation - Currently there is a reduction in throughput for classification. This will be updated in a future release. 8-bit GoogleNet runs at approximately 1300 img/sec.


- xfDNN-Preview-0.2a
	- Features:
		- 16-bit GoogLeNet v1
		- Caffe
		- Image Classification
		- DeepDetect REST Demo
		- DeepDetect Webcam Demo
	- Notes:
		- Intial Release 9/29/17

### Questions and Support

- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]
- [SDAccel Forums][]










[Xilinx AWS F1 xfDNN Test Drive here]: https://www.xilinx.com/applications/megatrends/machine-learning/aws-f1-test-drive.html
[Launching Instructions]: launching_instance.md
[Image Classification GoogLeNet v1 Demo]:image_classification.md
[DeepDetect REST Tutorial]:deepdetect_rest.md
[DeepDetect Webcam Tutorial]:deepdetect_webcam.md

[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
