# Prototxt to FPGA Tutorial

## Introduction
This example shows taking a single trained CAFFE model and prototxt for Googlenet-V1 and runs through all the steps to running it on the FPGA.  The first step is running the compiler to make sure the network is fully supported and fits in memory.  The second step is running the network on a calibration set for building the quantization data. For instructions on launching and connecting to instances, see [here][].  More in depth [compiler] and [quantizer] documentation is available.


## GoogLeNet-v1 Prototxt to FPGA Example

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
        # python tests/xfdnn_compiler.py -n /home/centos/googlenetv1.prototxt -s all -m 4 -i 28 -g /home/centos/network.cmd
	```

	When successful, the /home/centos/network.cmd and /home/centos/network.cmd.json will be created

	```

7. Navigate to `/xlnx/xfdnn_tools/quantize/`
	```
	# cd /xlnx/xfdnn_tools/quantize/
	```

7. The next step is running the quantizer on sample images from the network training or validation set

        network.cmd
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


8. Finally, navigate to the /opt/caffe directory and run the network through the run_network_8b.sh script:
cd /opt/caffe

# Works
run_network_8b.sh models/bvlc_googlenet/bvlc_googlenet_noLRN.dummydata.txt.xdnn.quantize dmaimodels/bvlc_googlenet/GoogleNetWithOutLRN.caffemodel /home/centos/network.cmd ./xdnn_scheduler/googlenet_quantized.json

# Segfault
run_network_8b.sh /home/centos/googlenetv1.prototxt models/bvlc_googlenet/GoogleNetWithOutLRN.caffemodel /home/centos/network.cmd ./xdnn_scheduler/googlenet_quantized.json

[here]: launching_instance.md
[compile]: compile.md
[quantize]: quantize.md