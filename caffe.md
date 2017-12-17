# Caffe Tutorial

## Introduction
This tutorial shows how to execute 8/16 bit networks through [Caffe][] with the included GoogLeNet-v1 and ResNet-50 models. Each of mode of models has been put into a run script that passes a few sample images to show accuracy and measure performance.

For instructions on launching and connecting to instances, see [here][].

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

3. Execute `./start_docker.sh` to enter application docker and enter the caffe directory
	```
	$ ./start_docker.sh
	/opt# cd caffe
	```

	In this directory you will see:
	- `run_googlenet_16b.sh` - This will run GoogLeNet-v1 with a 16b model.*
	- `run_googlenet_8b.sh`   - This will run GoogLeNet-v1 with a 8b model.*
	- `run_resnet_16b.sh`   - This will run ResNet-50 with a 16b model.*
	- `run_resnet_8b.sh`   - This will run Resnet-50 with a 8b model.*
	- `run_flowers_16b.sh`    - This Will run Flowers-102 with a 16b model.*
	- `run_places_16b.sh`    - This will run Places-365 with a 16b model.*

        \*Note: When running the test scripts, use sudo. AWS requires root (sudo) privileges to program the FPGA. For more details, [click here][].  The start_docker.sh script already starts the docker container with sudo so it is not needed in this tutorial.

4. Choose a script to run and execute with sudo:
	```
	/opt/caffe# ./run_googlenet_8b.sh
        ...
	[output truncated]
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

	At the end of the runs (output above), you can see the predictions of the four example images processed. The runtime from conv1/7x7_s2(Convolution) to pool5_7x7_s1 is reported for the batch of images however this is not the peak FPGA performance because this includes loading the weights and instructions ot the FPGA and this version of CAFFE does not have a multi-process pipeline to fully utilize the FPGA.  See the [Image Classification GoogLeNet-v1 Demo][] for a realistic performance demonstration.

	Another Example:
	```
	# ./run_flowers_16b.sh
	---------- Prediction 0 for examples/flowers/passion_00001.jpg ----------
	0.9541 - "love in the mist"
	0.0455 - "passion flower"
	0.0004 - "alpine sea holly"
	0.0000 - "spring crocus"
	0.0000 - "pink primrose"

	---------- Prediction 1 for examples/flowers/dahlia_03000.jpg ----------
	1.0000 - "pink-yellow dahlia"
	0.0000 - "bearded iris"
	0.0000 - "thorn apple"
	0.0000 - "rose"
	0.0000 - "barbeton daisy"

	---------- Prediction 2 for examples/flowers/windflower_06000.jpg ----------
	0.9940 - "windflower"
	0.0046 - "japanese anemone"
	0.0009 - "mexican aster"
	0.0004 - "bougainvillea"
	0.0001 - "columbine"

	---------- Prediction 3 for examples/flowers/lily_08000.jpg ----------
	0.9999 - "blackberry lily"
	0.0001 - "bird of paradise"
	0.0000 - "anthurium"
	0.0000 - "fritillary"
	0.0000 - "tiger lily"
	```

[Image Classification GoogLeNet-v1 Demo]:image_classification.md
[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
