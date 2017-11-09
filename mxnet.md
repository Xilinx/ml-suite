# MxNet Tutorial

## Introduction
This tutorial shows how to execute 8 bit networks through [MxNet][] with the included GoogLeNet v1 model. Each of mode of models has been put into a run script, that passes a few sample images, to show accuracy and measure performance.

For instructions on launching and connecting to instances, see [here][].

1. Connect to F1
2. Navigate to `/xfdnn_11_13_17/`
	```
	$ ls
caffe       docker_run_f1_mxnet.sh  imagenet      models  mxnet_docker  xlnx_docker
deepdetect  frameworks              imagenet_val  mxnet   xfdnn_tools


	```

3. Execute `./docker_run_f1_mxnet.sh` to enter application docker
	```
	$ ./docker_run_f1_mxnet.sh
	# 
	```

4. Navigate to

5. Choose a script to run and execute with sudo:
	```
	/opt/caffe$ sudo ./run_googlenet_8b.sh
	[output truncated]
	ANDBG googlenet runFpgaOptimized time: 2.25977 ms start: prob(Softmax) end:
	ANDBG googlenet runFpgaOptimized time: 62.4686 ms
	ANDBG total Forward time: 835.881 ms
	---------- Prediction 0 for examples/images/cat.jpg ----------
	0.9593 - "n02123159 tiger cat"
	0.0271 - "n02124075 Egyptian cat"
	0.0039 - "n02123045 tabby, tabby cat"
	0.0038 - "n02119789 kit fox, Vulpes macrotis"
	0.0018 - "n02326432 hare"

	---------- Prediction 1 for examples/images/cat_gray.jpg ----------
	0.4355 - "n02123394 Persian cat"
	0.1659 - "n02127052 lynx, catamount"
	0.1579 - "n02120079 Arctic fox, white fox, Alopex lagopus"
	0.0907 - "n02123159 tiger cat"
	0.0879 - "n02326432 hare"

	---------- Prediction 2 for examples/images/fish-bike.jpg ----------
	0.6006 - "n02797295 barrow, garden cart, lawn cart, wheelbarrow"
	0.3442 - "n04482393 tricycle, trike, velocipede"
	0.0165 - "n03785016 moped"
	0.0062 - "n03127747 crash helmet"
	0.0040 - "n02835271 bicycle-built-for-two, tandem bicycle, tandem"

	---------- Prediction 3 for examples/images/cat.jpg ----------
	0.9593 - "n02123159 tiger cat"
	0.0271 - "n02124075 Egyptian cat"
	0.0039 - "n02123045 tabby, tabby cat"
	0.0038 - "n02119789 kit fox, Vulpes macrotis"
	0.0018 - "n02326432 hare"
	```

	At the end of the runs (output above), you can see the predictions of the four example images processed. The speed of each run is reported as `runFpgaOptimized time: 2.25977 ms`.


[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[MxNet]:https://github.com/apache/incubator-mxnet
