# Prototxt to FPGA Tutorial

## Introduction
This example shows taking a single trained CAFFE model and prototxt for Googlenet-V1 and runs through all the steps to running it on the FPGA.  The first step is running the compiler to make sure the network is fully supported and fits in memory.  The second step is running the network on a calibration set for building the quantization data. For instructions on launching and connecting to instances, see [here][].  More in depth [compiler][] and [quantizer][] documentation is available.


## GoogLeNet-v1 Prototxt to FPGA Example

1. Connect to F1
2. Navigate to `/home/centos/xfdnn_18_03_19/caffe/`
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

6. This next command will execute GoogleNet-v1 compiler using a prototxt for CAFFE.  It will generate code for the xfDNN configuration available on the Xilinx Machine Learning Development Stack, Preview Edition.

	```
	# python tests/xfdnn_compiler.py -n /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt -s all -m 4 -i 28 -g /opt/caffe/network.cmd
	```

	When successful, the /opt/caffe/network.cmd file will be created.


7. Navigate to `/xlnx/xfdnn_tools/quantize/`
	```
	# cd /xlnx/xfdnn_tools/quantize/
	```

8. The next step is running the quantizer on sample images from the network training or validation set

	```
	# python quantize.pyc \
	--deploy_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
	--train_val_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_train_val.prototxt \
	--weights /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel \
	--quantized_train_val_model q.train_val.prototxt \
	--calibration_directory ../../imagenet_val/ \
	--calibration_size 8
	```

	When successful, the /opt/caffe/q.tran_val.json file will be created

9. Finally, navigate to the /opt/caffe directory and run the network on the FPGA through the run_network_8b.sh script:

	```
	# cd /opt/caffe
	#  run_network_8b.sh /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt models/bvlc_googlenet/GoogleNetWithOutLRN.caffemodel network.cmd q.train_val.json
	...
	---------- Prediction 0 for examples/images/cat.jpg ----------
	0.5036 - "n02123159 tiger cat"
	0.1366 - "n02124075 Egyptian cat"
	0.0581 - "n02123045 tabby, tabby cat"
	0.0402 - "n02119789 kit fox, Vulpes macrotis"
	0.0341 - "n02119022 red fox, Vulpes vulpes"
	...
	```

Note: As used here, the prototxt batch size must match the number of input images, 4 in this example


[here]: launching_instance.md
[compile]: compile.md
[quantize]: quantize.md
