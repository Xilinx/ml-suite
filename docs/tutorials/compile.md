# xfdnn Compiler Tutorial

## Introduction
The compiler script interfaces with different ML Frameworks, such as, Caffe and Tensorflow, to read a deep learning networks and generates a sequence of instructions for the xfDNN framework to execute on the FPGA.  This includes a computational graph traversal, node merging and optimization, memory allocation and, finally, instruction generation.

For instructions on launching and connecting to aws instances, see [here][].

## Support Features

For Each ML Framework, there is a different version of the complier in the `ml-suite/xfdnn/tools/compile/bin/` directory.
- Caffe - `xfdnn_compiler_caffe.pyc`
- Keras - `xfdnn_compiler_keras.pyc`
- MxNet - `xfdnn_compiler_mxnet.pyc`
- Tensforflow - `xfdnn_compiler_tensorflow.pyc`

Each of these tools have mostly the same arguments. As an example, `xfdnn_compiler_caffe` has the following arguments:

```
usage: xfdnn_compiler_caffe.py [-h] [-n NETWORKFILE] [-g GENERATEFILE]
                               [-w WEIGHTS] [-o PNGFILE] [-c CONCATSTRATEGY]
                               [-s STRATEGY] [--schedulefile SCHEDULEFILE]
                               [-i DSP] [-v] [-m MEMORY] [-d DDR] [-p PHASE]
                               [-r RANKDIR]
```				 
- `-h, --help`- shows this help message and exit
- `-n, --networkfile` - NETWORKFILE Input prototxt for compiler
- `-g, --generatefile` - GENERATEFILE Output of Compiler, xDNN instructions
- `-w, --weights` - WEIGHTS Output of weights for use with deployment APIs
- `-o , --pngfile` - PNGFILE Outputs Optimized Graph as a PNG file, Requires dot executable
- `-c, --concatstrategy` - CONCATSTRATEGY
- `-s, --strategy` - STRATEGY Strategies for compiler (default: all)
- `--schedulefile`- SCHEDULEFILE Show layer by layer memory layout  (optional)
-  `-i, --dsp` DSP xdnn kernel dimension (28 or 56, default: 28) 28 is also known as "Med" and 56 as "Large"
-  `-v, --verbose`
- `-m, --memory MEMORY` - On-chip Memory available in MB (default: 4). Needs to match intended overlaybin.
- `-d, --ddr DDR` - Users DDR memory along with on-chip memory. If the network is too large for the device, set this to `256`.
- `-p , --phase PHASE`- Caffe prototxt phase (TRAIN, TEST, ALL default: TEST)
- `-r, --rankdir RANKDIR`

Any argument that is not passed will be set with a default value.

### GoogLeNet v1 Example

1. Connect to F1 or start your anaconda environment
2. Navigate to `ml-suite/xfdnn/tools/compile/bin`
3. This next command will execute GoogleNet-v1 compiler using a prototxt for CAFFE.  It will generate code for the xfDNN configuration available on the Xilinx Machine Learning Development Stack, Preview Edition
	```
	# python tests/xfdnn_compiler_caffe.py -n /ml-suite/models/caffe/bvlc_googlenet_quantized/GoogleNetWithOutLRN_dummydata_deploy.prototxt \
	-s all \
	-m 4 \
	-i 28 \
	-g network.cmd \
	-w /ml-suite/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel
	```

	In the console output the compiler will try various strategies and when successful will report the minimum memory:

	```
	Allocating Memory
	Trying strategy bysize
	Minimum Memory 133 ['inception_5b/pool'] 4194304.0
	```

	If the memory provided is not sufficient no strategy will produce a successful compilation:

	```
	Allocating Memory
	Trying strategy bysize
	Trying strategy bottom
	Trying strategy top
	Trying strategy tops
	Trying strategy bottle
	Trying strategy bottles
	Trying strategy xXx
	Trying strategy shuffle
	```

4. The output that may be passed to xfDNN by the scripts as shown in the Caffe tutorial is the network.cmd file located in the run directory.  The network.cmd.json includes meta-data about the network as well as a list of unsupported layers which will not be run on the FPGA.

	network.cmd:
	```
	# # SKIPPED data [u'Input'] ['layer'] data: type=Input, sizes=None, shapes=None, sched 0 Kernel None Strides None Padding None  NO VALID CODE
	2 XNConv conv1/7x7_s2 7 2 16 26 2 1 1 0x0 224 3 0x70000 112 64
	# FUSED "conv1/relu_7x7" [u'ReLU'] ['inplace_layer']
	4 XNMaxPool pool1/3x3_s2 3 2 0 0x70000 112 64 0x0 56
	5 XNConv conv2/3x3_reduce 1 1 16 26 2 1 1 0x0 56 64 0x70000 56 64
	# FUSED "conv2/relu_3x3_reduce" [u'ReLU'] ['inplace_layer']
	7 XNConv conv2/3x3 3 1 16 26 2 1 1 0x70000 56 64 0xe0000 56 192
	# FUSED "conv2/relu_3x3" [u'ReLU'] ['inplace_layer']
	9 XNMaxPool pool2/3x3_s2 3 2 0 0xe0000 56 192 0x0 28
	...
	```

[here]: docs/tutorials/launching_instance.md
