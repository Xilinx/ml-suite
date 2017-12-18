# Compiler Tutorial

## Introduction
The compiler script interfaces with CAFFE to read a deep learning model and generates a sequence of instructions for the xfDNN framework to execute on the FPGA.  This includes a computational graph traversal, node merging and optimization, memory allocation and, finally, instruction generation.

For instructions on launching and connecting to aws instances, see [here][].

## Support Features

To run the compiler, use the command `python xfdnn_compiler.pyc` with the instructions below.

List of Arguments available:

- `[-h]`
- `[-n,--networkfile DEPLOY_MODEL]` - Input prototxt for compiler
- `[-s,--strategy STRATEGY]` - Strategies for compiler (default: all)
- `[-m,--memory MEMSIZE]` - On-chip Memory available in MB (default: 4)
- `[-w,--weights WEIGHTS]` -  Caffe trained model (optional)
- `[-g,--generatecode FILENAME]` -  Output commands in XFDNN test and json
- `[-o,--pngfile FILENAME]` -  Output png file of graph read by compiler (optional)
- `[-q,--quantization FILENAME]` -  Include quantizer output during compiler (optional)
- `[--schedule FILENAME]` - Show layer by layer memory layout  (optional)
- `[-i,--dsp DSP_DIMENSION]` - xfDNN kernel dimension (28 or 56, default: 28)
- `[-p,--phase PHASE]` - Caffe prototxt phase (TRAIN, TEST, ALL default: TEST)

Any argument that is not passed will be set with a default value.

## GoogLeNet v1 Example

1. Connect to F1
2. Navigate to `/home/centos/xfdnn_17_12_15/caffe/`
	```
	$ ls
	classification.bin  kernelSxdnn_hw_f1_16b.xclbin  run_common.sh         run_places_16b.sh  xdnn_scheduler
	data                kernelSxdnn_hw_f1_8b.xclbin   run_cpu_env.sh        run_resnet_16b.sh  xlnx-docker
	demo                libs                          run_flowers_16b.sh    run_resnet_8b.sh   xlnx-xdnn-f1
	examples            models                        run_googlenet_16b.sh  sdaccel.ini
	execDocker.sh       README                        run_googlenet_8b.sh   start_docker.sh
	```

3. Execute `./start_docker.sh` to enter application docker
	```
	$ ./start_docker.sh
	/opt#
	```

4. Set XFDNN_ROOT to /xlnx
	```
	export XFDNN_ROOT=/xlnx
	```

5. Navigate to `/xlnx/xfdnn_tools/compile/`
	```
	# cd /xlnx/xfdnn_tools/compile/ 	
	```

6. This next command will execute GoogleNet-v1 compiler using a prototxt for CAFFE.  It will generate code for the XFDNN configuration available on the Xilinx Machine Learning Development Stack, Preview Edition
	```
  # python tests/xfdnn_compiler.py -n /xlnx/xfdnn_tools/models/caffe/bvlc_googlenet_quantized/GoogleNetWithOutLRN_dummydata_deploy.prototxt -s all -m 4 -i 28 -g network.cmd
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

7. The output that may be passed to xfDNN by the scripts as shown in the Caffe tutorial is the network.cmd file located in the run directory.  The network.cmd.json includes meta-data about the network as well as a list of unsupported layers which will not be run on the FPGA.

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

[here]: launching_instance.md
