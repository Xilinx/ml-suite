# Machine Learning Development Stack From Xilinx, Preview Edition

### Tutorials
- [Image Classification GoogLeNet-v1 Demo][]
- [DeepDetect REST Tutorial][]
- [DeepDetect Webcam Tutorial][]
- [Caffe Tutorial][]
- [MxNet Tutorial][]
- [Quantization Tutorial][]
- [Compiler Tutorial][]
- [Prototxt to FPGA Tutorial][]
- [Pyhton API Tutorials][]




### Release Notes

- xfdnn_18_03_15
	- Updates ...

- xfdnn_17_12_15
	- General script and documentation cleanup
	- Added Compiler Tutorial
	- Added End to End Tutorial
	- Known Issues:


- xfdnn_17_11_13.1
	- Supported Frameworks:
		- Caffe
		- MxNet
	- Models
		- 8/16-bit GoogLeNet v1
		- 8/16-bit ResNet50
		- 8/16-bit Flowers102
		- 8/16-bit Places365
	- Examples
		- DeepDetect REST Tutorial
		- DeepDetect Webcam
		- Caffe Image Classification
		- MxNet Image Classification
		- x8 FPGA Pooling GoogLeNet v1 Demo
	- xfDNN Tools
		- Compiler
		- Quantizer
	- Known Issues:
		- libdc1394 error: Failed to initialize libdc1394
			- Some of the examples will report this error, but it can be ignored.

### Questions and Support

- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]
- [SDAccel Forums][]

[Image Classification GoogLeNet-v1 Demo]:image_classification.md
[DeepDetect REST Tutorial]:deepdetect_rest.md
[DeepDetect Webcam Tutorial]:deepdetect_webcam.md
[Quantization Tutorial]:quantize.md
[Compiler Tutorial]:compile.md
[Prototxt to FPGA Tutorial]:endtoend.md
[Caffe Tutorial]:caffe.md
[MxNet Tutorial]:mxnet.md
[Pyhton API Tutorials]: pythonexample.md

[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
