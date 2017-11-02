# Image Classification GoogLeNet v1 Demo

## Introduction
Image classification is one of the most common benchmarks for machine learning. This tutorial shows you how to launch the image classification GoogLeNet v1 demo from the Test Drive environment. Once the demo is started, you will be able to view the demo and monitor demo performance from any internet connected web browser.


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

		For more information on running the GoogLeNet/ResNet50 scripts view the [Running 8/16 bit Networks][] tutorial.

4. Execute the `./run_demo.sh` script to start the demo
	```
	/opt/caffe_ristretto$ ./run_demo.sh
	Starting demo...
	kill: usage: kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]
	Starting producer...
	Starting Web Interface
	```
	Start up of the demo will take a few minutes, but once its complete, the console will start displaying numbers.

5. From your host machine: the demo will display at the following web address:
	"http://yourpublicdns.compute-1.amazonaws.com:8998/static/www/xdnn1.html"

	To get your 'yourpublicdns.compute-1.amazonaws.com' refer to the launching and connecting instructions
	From your browser you will see the running Image Classification Demo:

	![](img/image_classification.png)


[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[Running 8/16 bit Networks]: classification_16-8b.md
