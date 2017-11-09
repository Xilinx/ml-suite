# MxNet Tutorial

## Introduction
This tutorial shows how to execute 8 bit networks through [MxNet][] with the included GoogLeNet v1 model. Each of mode of models has been put into a run script, that passes a few sample images, to show accuracy and measure performance.

For instructions on launching and connecting to instances, see [here][].

1. Connect to F1
2. Navigate to `/xfdnn_17_11_13/mxnet/`
	```
	$ cd /xfdnn_17_11_13/mxnet/
	$ ls
	amalgamation     cpp-package             include                        MKL_README.md   ps-lite          tests
	appveyor.yml     cub                     Jenkinsfile                    models          python           tools
	benchmark        DISCLAIMER              KEYS                           mshadow         README.md        xclbin
	bin              dlpack                  lib                            mxnet_docker    readthedocs.yml  xfdnn_env
	build            dmlc-core               LICENSE                        NEWS.md         R-package        xlnx_docker
	cmake            docker                  make                           nnvm            scala-package    xlnx_lib
	CMakeLists.txt   docker_build_mxnet.sh   Makefile                       NOTICE          setup-utils      xlnx_rt_xdnn
	CODEOWNERS       docker_run_f1_mxnet.sh  matlab                         perl-package    snapcraft.yaml
	config.mk        docs                    mkl                            plugin          snap.python
	CONTRIBUTORS.md  example                 mklml_lnx_2018.0.20170720.tgz  prepare_mkl.sh  src
	```

3. Execute `./docker_run_f1_mxnet.sh` to enter application docker and navigate to `/mxnet/examples/xlnx/simiple-image-classify/`
	```
	$ ./docker_run_f1_mxnet.sh
	# cd example/xlnx/simple-image-classify/
	$ ls
	beagle.jpg  classify_fpga.sh  sdaccel_profile_summary.csv   synset.txt
	cat.jpg     classify.py       sdaccel_profile_summary.html  wolf.jpg
	```

4. Here is a example of running a single image with the model GoogLeNet v1. Run the `./classify_fpga.sh` script and pass one of the example images here for it to classify.
	```
	# ./classify_fpga.sh beagle.jpg
	XBLAS # FPGAs: 1
	[XBLAS] # kernels: 1
	[XDNN] using custom DDR banks 0,2,1,1
	Device/Slot[0] (/dev/xdma0, 0:0:1d.0)
	xclProbe found 1 FPGA slots with XDMA driver running
	CL_PLATFORM_VENDOR Xilinx
	CL_PLATFORM_NAME Xilinx
	CL_DEVICE_0: 0x3898a20
	CL_DEVICES_FOUND 1, using 0
	loading /opt/mxnet/xclbin/kernelSxdnn_hw_f1_16b.xclbin
	[XBLAS] kernel0: kernelSxdnn_0
	XBLAS online! (d=0)
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	 -------------------
	probability=0.805006, class=n02088364 beagle
	probability=0.109553, class=n02089867 Walker hound, Walker foxhound
	probability=0.054156, class=n02089973 English foxhound
	probability=0.003512, class=n02101388 Brittany spaniel
	probability=0.002501, class=n02088632 bluetick
	```

5. To see the agreggated accuracy of GoogLeNet v1, navigate to `/mxnet/example/image-classification/`. Execute the example, `./score_fpga.sh`
	```
	# cd mxnet/example/image-classification/
	# ls
	README.md	    score.pyc			    symbol_inception-resnet-v1.R  test_score.py
	__init__.py	    score_fpga.py		    symbol_inception-resnet-v2.R  train_cifar10.R
	benchmark.py	    score_fpga.sh		    symbol_lenet.R		  train_cifar10.py
	benchmark_score.py  sdaccel_profile_summary.csv     symbol_mlp.R		  train_imagenet.R
	common		    sdaccel_profile_summary.html    symbol_resnet-28-small.R	  train_imagenet.py
	data		    symbol_alexnet.R		    symbol_resnet-v2.R		  train_mnist.R
	fine-tune.py	    symbol_googlenet.R		    symbol_resnet.R		  train_mnist.py
	predict-cpp	    symbol_inception-bn-28-small.R  symbol_vgg.R		  train_model.R
	score.py	    symbol_inception-bn.R	    symbols

	# ./score_fpga.sh
	Downloading dataset ...
	INFO:root:data/val_256_q90.rec exists, skipping download
	[06:17:30] src/io/iter_image_recordio_2.cc:153: ImageRecordIOParser2: data/val_256_q90.rec, use 3 threads for decoding..
	XBLAS # FPGAs: 1
	[XBLAS] # kernels: 1
	[XDNN] using custom DDR banks 0,2,1,1
	Device/Slot[0] (/dev/xdma0, 0:0:1d.0)
	xclProbe found 1 FPGA slots with XDMA driver running
	CL_PLATFORM_VENDOR Xilinx
	CL_PLATFORM_NAME Xilinx
	CL_DEVICE_0: 0x231a590
	CL_DEVICES_FOUND 1, using 0
	loading /opt/mxnet/xclbin/kernelSxdnn_hw_f1_16b.xclbin
	[XBLAS] kernel0: kernelSxdnn_0
	XBLAS online! (d=0)
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	INFO:root:('accuracy', 0.65239043824701193)
	INFO:root:('top_k_accuracy_5', 0.8635458167330677)
	```

	At the end of the run, the top accuracy and top 5 accurary is listed:
	```
	INFO:root:('accuracy', 0.65239043824701193)
	INFO:root:('top_k_accuracy_5', 0.8635458167330677)
	```



[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[MxNet]:https://github.com/apache/incubator-mxnet
