# Using xfDNN Python APIs Tutorial


## Introduction
This tutorial shows how to execute DNNs on the FPGA using our Python xfDNN API. We have included the GoogLeNet v1 and Resnet-50 models. We provide two examples of applications using the Python xfDNN API:

	1) A batch classification example that streams images from disk through the FPGA for classification.
	2) A Multi-Process example that shows different DNNs running on different PEs (cores) on the FPGA.

Directory overview of this tutorial:
```
pyxdnn/
└── examples
    ├── batch_classify
    │   ├── batch_classify.py
    │   ├── images
    │   ├── images_224
    │   ├── run.sh
    │   └── synset_words.txt
    ├── common
    └── multinet
        ├── mytest.json
        ├── run.sh
        ├── synset_words.txt
        └── test_classify_async_multinet.py
```

### Batch Classification (Streaming)

For instructions on launching and connecting to aws instances, see [here][].

1. Connect to F1
2. Navigate to `/home/centos/xfdnn_18_03_19/pyxdnn/examples/batch_classify/`
	```
	$ cd xfdnn_18_03_19/pyxdnn/examples/batch_classify/
	$ ls
	batch_classify.py  images  images_224  run.sh  synset_words.txt
	```
3. Run `sudo run.sh` to execute the example.
	```
	$ sudo ./run.sh
	=============== pyXDNN =============================
	[XBLAS] # kernels: 1
	[XDNN] using custom DDR banks 0,2,1,1
	[time] loadImageBlobFromFile/OpenCV (4.72 ms):
	[time] loadImageBlobFromFile/OpenCV (3.29 ms):
	[time] loadImageBlobFromFile/OpenCV (2.80 ms):
	[time] loadImageBlobFromFile/OpenCV (1.98 ms):
	[time] loadImageBlobFromFile/OpenCV (2.13 ms):
	[time] loadImageBlobFromFile/OpenCV (2.46 ms):
	[time] loadImageBlobFromFile/OpenCV (1.61 ms):
	[time] loadImageBlobFromFile/OpenCV (2.42 ms):
	[time] prepareImages (28.91 ms):
	Device/Slot[0] (/dev/xdma0, 0:0:1d.0)
	xclProbe found 1 FPGA slots with XDMA driver running
	CL_PLATFORM_VENDOR Xilinx
	CL_PLATFORM_NAME Xilinx
	CL_DEVICE_0: 0x15e4680
	CL_DEVICES_FOUND 1, using 0
	loading ../common/kernel.xclbin
	[XBLAS] kernel0: kernelSxdnn_0
	Loading weights/bias/quant_params to FPGA...
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	WARNING: unaligned host pointer detected, this leads to extra memcpy
	[XDNN] FPGA metrics (0/0/0)
	[XDNN]   write_to_fpga  : 0.16 ms
	[XDNN]   exec_xdnn      : 24.25 ms
	[XDNN]   read_from_fpga : 0.08 ms
	[XDNN] FPGA metrics (0/1/0)
	[XDNN]   write_to_fpga  : 0.15 ms
	[XDNN]   exec_xdnn      : 33.32 ms
	[XDNN]   read_from_fpga : 0.08 ms
	[XDNN] FPGA metrics (0/2/0)
	[XDNN]   write_to_fpga  : 0.08 ms
	[XDNN]   exec_xdnn      : 33.20 ms
	[XDNN]   read_from_fpga : 0.06 ms
	[XDNN] FPGA metrics (0/3/0)
	[XDNN]   write_to_fpga  : 0.15 ms
	[XDNN]   exec_xdnn      : 38.91 ms
	[XDNN]   read_from_fpga : 0.06 ms
	[time] FPGA xdnn execute (527.82 ms):
	[time] FC (2.03 ms):

	---------- Prediction 0 for images_224/cat gray.jpg ----------
	0.3433 - "n02123394 Persian cat"
	0.1283 - "n02127052 lynx, catamount"
	0.0940 - "n02124075 Egyptian cat"
	0.0520 - "n02123159 tiger cat"
	0.0520 - "n02123597 Siamese cat, Siamese"

	---------- Prediction 1 for images_224/cat.jpg ----------
	0.4036 - "n02123159 tiger cat"
	0.2442 - "n02124075 Egyptian cat"
	0.1219 - "n02123045 tabby, tabby cat"
	0.0587 - "n02127052 lynx, catamount"
	0.0368 - "n02123394 Persian cat"

	---------- Prediction 2 for images_224/ILSVRC2012_val_00003225.JPEG ----------
	0.9960 - "n02422106 hartebeest"
	0.0023 - "n02422699 impala, Aepyceros melampus"
	0.0014 - "n02423022 gazelle"
	0.0002 - "n02408429 water buffalo, water ox, Asiatic buffalo, Bubalus bubalis"
	0.0001 - "n02417914 ibex, Capra ibex"

	---------- Prediction 3 for images_224/ILSVRC2012_val_00001172.JPEG ----------
	0.3800 - "n02114367 timber wolf, grey wolf, gray wolf, Canis lupus"
	0.2708 - "n02114548 white wolf, Arctic wolf, Canis lupus tundrarum"
	0.1437 - "n02109961 Eskimo dog, husky"
	0.1004 - "n02110063 malamute, malemute, Alaskan malamute"
	0.0573 - "n02110185 Siberian husky"

	---------- Prediction 4 for images_224/fish-bike.jpg ----------
	0.4856 - "n02797295 barrow, garden cart, lawn cart, wheelbarrow"
	0.1152 - "n04482393 tricycle, trike, velocipede"
	0.0570 - "n04258138 solar dish, solar collector, solar furnace"
	0.0211 - "n02835271 bicycle-built-for-two, tandem bicycle, tandem"
	0.0179 - "n04044716 radio telescope, radio reflector"

	---------- Prediction 5 for images_224/cat_gray.jpg ----------
	0.3433 - "n02123394 Persian cat"
	0.1283 - "n02127052 lynx, catamount"
	0.0940 - "n02124075 Egyptian cat"
	0.0520 - "n02123159 tiger cat"
	0.0520 - "n02123597 Siamese cat, Siamese"

	---------- Prediction 6 for images_224/cat gray.jpg ----------
	0.3433 - "n02123394 Persian cat"
	0.1283 - "n02127052 lynx, catamount"
	0.0940 - "n02124075 Egyptian cat"
	0.0520 - "n02123159 tiger cat"
	0.0520 - "n02123597 Siamese cat, Siamese"

	---------- Prediction 7 for images_224/cat.jpg ----------
	0.4036 - "n02123159 tiger cat"
	0.2442 - "n02124075 Egyptian cat"
	0.1219 - "n02123045 tabby, tabby cat"
	0.0587 - "n02127052 lynx, catamount"
	0.0368 - "n02123394 Persian cat"

	Num processed: 8

	[time] Total loop (21915.48 ms)
	```
	This example runs 8 images through the Python API in batch or streaming mode.

	The `run.sh` example calls the API, and takes the following parameters:
	batch_classify.py
	- `--xclbin` 		- Defines which FPGA binary to use. By Default, leave set to the binary in the example
	- `--xlnxcfg` 	- FPGA config file
	- `--xlnxnet` 	- FPGA instructions generated by xfDNN Compiler for the network
	- `--fpgaoutsz`	- Size of FPGA output blob
	- `--datadir`		- Path to data files to run for the network (weights)
	- `--labels`		- Result -> labels translation file (typically Text File)
	- `--xlnxlib`		- FPGA xfDNN lib
	- `--imagedir`	- Directory with image files to classify
	- `--useblas`		- Use BLAS-optimized functions (requires xfDNN lib compiled with BLAS)

4. You can try different networks and precision types and examples for these are in `run.sh`. By default, `Googlenet v1 16-bit` is executed, but you can comment out that option and uncomment the other options.

```
	# Run Googlenet v1 16-bit
	/usr/bin/python batch_classify.py \
	--xclbin $PYXDNN_ROOT/kernel.xclbin \
	--xlnxnet $PYXDNN_ROOT/xdnn_scheduler/googlenet.fpgaaddr.64.txt \
	--fpgaoutsz 1024 \
	--datadir $PYXDNN_ROOT/data_googlenet_v1 \
	--labels synset_words.txt \
	--xlnxlib $PYXDNN_ROOT/lib/libxblas.so \
	--imagedir images_224 \
	--useblas

	# Run Googlenet v1 8-bit
	#/usr/bin/python batch_classify.py \
	--xclbin $PYXDNN_ROOT/kernel_8b.xclbin \
	--xlnxnet $PYXDNN_ROOT/xdnn_scheduler/googlenet.fpgaaddr.64.txt \
	--fpgaoutsz 1024 \
	--datadir $PYXDNN_ROOT/data_googlenet_v1 \
	--labels synset_words.txt \
	--xlnxlib $PYXDNN_ROOT/lib/libxblas.so \
	--imagedir images_224 \
	--useblas \
	--xlnxcfg $PYXDNN_ROOT/xdnn_scheduler/googlenet_quantized.json


	# Run Resnet-50 16-bit
	#/usr/bin/python batch_classify.py \
	--xclbin $PYXDNN_ROOT/kernel.xclbin \
	--xlnxnet $PYXDNN_ROOT/xdnn_scheduler/resnet.fpgaaddr.64.txt \
	--fpgaoutsz 2048 \
	--datadir $PYXDNN_ROOT/data_resnet_50 \
	--labels synset_words.txt \
	--xlnxlib $PYXDNN_ROOT/lib/libxblas.so \
	--imagedir images_224 \
	--useblas

	# Run Resnet-50 8-bit
	#/usr/bin/python batch_classify.py \
	--xclbin $PYXDNN_ROOT/kernel_8b.xclbin \
	--xlnxnet $PYXDNN_ROOT/xdnn_scheduler/resnet.fpgaaddr.64.txt \
	--fpgaoutsz 2048 \
	--datadir $PYXDNN_ROOT/data_resnet_50 \
	--labels synset_words.txt \
	--xlnxlib $PYXDNN_ROOT/lib/libxblas.so \
	--imagedir images_224 \
	--useblas \
	--firstfpgalayer conv1 \
	--xlnxcfg $PYXDNN_ROOT/xdnn_scheduler/resnet_quantized.json
```

### Multi-Process Classification

For instructions on launching and connecting to aws instances, see [here][].

1. Connect to F1
2. Navigate to `/home/centos/xfdnn_18_03_19/pyxdnn/examples/batch_classify/`
	```
	$ cd xfdnn_18_03_19/pyxdnn/examples/multinet/
	$ ls
	mytest.json  run.sh  synset_words.txt  test_classify_async_multinet.py
	```
3. Run `sudo run.sh` to execute the example.
	```
	$ sudo ./run.sh
		=============== pyXDNN Async MultiNet =============================
		[XBLAS] # kernels: 1
		[XDNN] using custom DDR banks 0,2,1,1
		Device/Slot[0] (/dev/xdma0, 0:0:1d.0)
		xclProbe found 1 FPGA slots with XDMA driver running
		CL_PLATFORM_VENDOR Xilinx
		CL_PLATFORM_NAME Xilinx
		CL_DEVICE_0: 0x2940c70
		CL_DEVICES_FOUND 1, using 0
		loading ../common/kernel.xclbin
		[XBLAS] kernel0: kernelSxdnn_0

		After createHandle (7039.839983 ms):
		Loading weights/bias/quant_params to FPGA...
		Loading weights/bias/quant_params to FPGA...

		After init (43211.190939 ms):
		WARNING: unaligned host pointer detected, this leads to extra memcpy
		WARNING: unaligned host pointer detected, this leads to extra memcpy
		[XDNN] FPGA metrics (0/0/0)
		[XDNN]   write_to_fpga  : 0.15 ms
		[XDNN]   exec_xdnn      : 10.42 ms
		[XDNN]   read_from_fpga : 0.06 ms
		WARNING: unaligned host pointer detected, this leads to extra memcpy
		WARNING: unaligned host pointer detected, this leads to extra memcpy

		After Execonly (361.441135 ms):
		[XDNN] FPGA metrics (0/1/0)
		[XDNN]   write_to_fpga  : 0.15 ms
		[XDNN]   exec_xdnn      : 19.67 ms
		[XDNN]   read_from_fpga : 0.06 ms

		After wait (20.205975 ms):

		After FC (1.589060 ms):

		After Softmax (0.423908 ms):

		---------- Prediction 0 ----------
		0.3706 - "n02123159 tiger cat"
		0.2542 - "n02124075 Egyptian cat"
		0.1062 - "n02123045 tabby, tabby cat"
		0.0480 - "n02127052 lynx, catamount"
		0.0349 - "n02123394 Persian cat"

		After FC (111.846924 ms):

		After Softmax (0.396013 ms):

		---------- Prediction 0 ----------
		0.3498 - "n02113023 Pembroke, Pembroke Welsh corgi"
		0.1936 - "n02124075 Egyptian cat"
		0.1211 - "n02119789 kit fox, Vulpes macrotis"
		0.0942 - "n02119022 red fox, Vulpes vulpes"
		0.0835 - "n02123159 tiger cat"

		Success!
	```
	This example runs through two networks/models, GoogleNetv1 and ResNet50. The networks and parameters are defined in the json input file. This example uses `mytest.json`:
	```
	{
		"confs":[
		  {
		    "name":"googlenet_1",
		    "net":"googlenet",
		    "datadir":"../common/data_googlenet_v1",
		    "netcfg":"../common/xdnn_scheduler/googlenet.fpgaaddr.64.txt",
		    "PE":"0",
		    "firstFpgaLayerName":"conv1/7x7_s2",
		    "fpgaoutsz":"1024",
		    "useblas": true
		    },
		  {
		    "name":"resnet_2",
		    "net":"resnet",
		    "datadir":"../common/data_resnet_50",
		    "netcfg":"../common/xdnn_scheduler/resnet.fpgaaddr.64.txt",
		    "PE":"1",
		    "firstFpgaLayerName":"conv1",
		    "fpgaoutsz":"2048",
		    "useblas": true
		  }
		]
	}
	```
	The parameters are defined in a similar way to the batch example, but `PE: `defines which xDNN processing engine on the FPGA you would like to run this network. The number of xDNN processing engines available on the FPGA is dependent the xclbin selected, which can be 1, 2 and 4. The default xclbin used in these examples is4 xDNN processing engines.
	This example runs Googlenetv1 on PE:0 and Resnet50 on PE:1.




[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
