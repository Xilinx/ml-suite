# Image Classification GoogLeNet v1 Demo

## Introduction
Image classification is one of the most common benchmarks for machine learning. This tutorial shows you how to launch the image classification GoogLeNet v1 8-bit demo from the Test Drive environment. Once the demo is started, you will be able to view the demo and monitor demo performance from any internet connected web browser.


For instructions on launching and connecting to instances, see [here][].

1. Connect to F1
2. Navigate to `/xfdnn_17_11_13/caffe/`
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

3. Execute `./startDocker.sh` to enter application docker and navigate to `/caffe/demo/`
	```
	$ ./startDocker.sh
	# cd caffe/demo/
	# ls
	kill_demo.sh  run_demo_gui.sh      run_mp_fc_xdnn.sh  sdaccel.ini                  sdaccel_profile_summary.html
	run_demo.sh   run_mp_conv_xdnn.sh  run_mp_fpga_flow   sdaccel_profile_summary.csv  servergui
	```

4. Execute the `./run_demo.sh` script to start the demo
	```
	# ./run_demo.sh
	Starting demo...
	kill: usage: kill [-s sigspec | -n signum | -sigspec] pid | jobspec ... or kill -l [sigspec]
	Starting producer...
	Starting Web Interface
	```
	Start up of the demo will take a few minutes, but once it's complete, the console will start displaying numbers.

5. From your host machine: the demo will display at the following web address:
	"http://yourpublicdns.compute-1.amazonaws.com:8998/static/www/xdnn1.html"

	To get your 'yourpublicdns.compute-1.amazonaws.com', refer to the launching and connecting instructions.
	From your browser you will see the running Image Classification Demo:

	![](img/image_classification.png)

6. To stop the demo, type `ctrl + c`

[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[Running 8/16 bit Networks]: classification_16-8b.md
