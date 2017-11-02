# Running 8/16 bit Networks

## Introduction
This tutorial shows how to execute 8/16 bit networks with the included GoogLeNet v1 and ResNet50 models. Each of mode of models has been put into a run script, that passes a few sample images, to show accuracy and measure performance.

For instructions on launching and connecting to instances, see [here][].

1. Connect to F1
2. Navigate to `/xfdnn_testdrive/caffe/`
	```
	$ ls
	classification.bin            libs                  run_mp_conv_xdnn.sh           servergui
	data                          models                run_mp_fc_xdnn.sh             start_caffe_docker.sh
	examples                      run_common.sh         run_mp_fpga_flow              xdnn_scheduler
	exec_caffe_docker.sh          run_cpu_env.sh        run_resnet_16b.sh             xlnx-docker
	imagenet                      run_demo_gui.sh       run_resnet_8b.sh              xlnx-xdnn-f1
	kernelSxdnn_hw_f1_16b.xclbin  run_demo.sh           sdaccel.ini
	kernelSxdnn_hw_f1_8b.xclbin   run_googlenet_16b.sh  sdaccel_profile_summary.csv
	kill_demo.sh                  run_googlenet_8b.sh   sdaccel_profile_summary.html
	```

3. Execute `./start_caffe_docker.sh` to enter application docker
	```
	$ ./start_caffe_docker.sh
	/opt/caffe_ristretto$
	```
	In this directory you will see:
    - `run_googlenet_16b.sh` - This will run GoogLeNet with a 16b model.*
    - `run_googlenet_8b.sh`   - This will run GoogLeNet with a 8b model.*
		- `run_resnet_16b.sh`   - This will run ResNet50 with a 16b model.*
		- `run_resnet_8b.sh`   - This will run Resnet50 with a 8b model.*
    - `run _demo.sh`    - Will run a Image Classification Speed of GoogLeNet v1 demo.
    - `kill_demo.sh`    - Will kill the Image Classification .

		\*Note: When running the test scripts, use sudo. AWS requires root (sudo) privileges to program the fpga. For more details, [click here][].


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
