# Quantization Tutorial

## Introduction
The quantize script produces an optimal target quantization within a matter of minutes from a given network (prototxt and caffemodel) and calibration set (unlabeled input images) without requiring hours of retraining or a labeled dataset.

For instructions on launching and connecting to instances, see [here][].

## Support Features

To run the quantizer, use the command `python quantize.pyc` with the instructions below.

List of Arguments available:

- `[-h]`
- `[--deploy_model DEPLOY_MODEL]` - Input prototxt for calibration
- `[--train_val_model TRAIN_VAL_MODEL]` - Input training prototxt for calibration
- `[--weights WEIGHTS]` - FP32 pretrained caffe model
- `[--quantized_train_val_model QUANTIZED_TRAIN_VAL_MODEL]` - Output file name for calibration
- `[--calibration_directory CALIBRATION_DIRECTORY]` - Dir of dataset of original images
- `[--calibration_size CALIBRATION_SIZE]` - Number of images to use for calibration, default is 8
- `[--bitwidths BITWIDTHS] ` - Bit widths for input,params,output default: 8,8,8
- `[--dims DIMS]`            - Dimensions for first layer, default 3,224,224
- `[--transpose TRANSPOSE] ` - Passed to caffe.io.Transformer function set_transpose, default 2,0,1
- `[--channel_swap CHANNEL_SWAP]` - Passed to caffe.io.Transformer function set_channel_swap, default 2,1,0
- `[--raw_scale RAW_SCALE] ` - Passed to caffe.io.Transformer function set_raw_scale, default 255.0
- `[--mean_value MEAN_VALUE]` - Passed to caffe.io.Transformer function set_mean, default 104,117,123

Any argument that is not passed will be set with a default value for the provided Googlenet-v1 example.

## GoogLeNet v1 Example

1. Connect to F1
2. Navigate to `/home/centos/xfdnn_17_12_15/caffe/`.</br>
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

4. Navigate to `/xlnx/xfdnn_tools/quantize/`
	```
	# cd /xlnx/xfdnn_tools/quantize/
	```

5. This next command will execute GoogleNet-v1 quantization using deploy and train/validation models provided in the /xlnx/models directory.  This quantization expects at least 8 images to be available in the `/home/centos/xfdnn_17_12_15/imagenet_val` directory.  Refer to http://www.image-net.org/download-imageurls for downloading ILSVRC files from ImageNet.  Other files may be used and do not require any special file naming convention.
	```
	# python quantize.pyc \
	--deploy_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
	--train_val_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_train_val.prototxt \
	--weights /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel \
	--quantized_train_val_model q.train_val.prototxt \
	--calibration_directory ../../imagenet_val/ \
	--calibration_size 8
	```

        This translates into the following options:

	```
  # python quantize.pyc --deploy_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt --train_val_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_train_val.prototxt --weights /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel --quantized_train_val_model q.train_val.prototxt --calibration_directory ../../imagenet_val/ --calibration_size 8 --bitwidths 8,8,8 --dims 3,224,224 --transpose 2,0,1 --channel_swap 2,1,0 --raw_scale 255.0 --mean_value 104,117,123 --input_scale 1.0
	 ```

	To run on more than 8 images (recommended is at least 20), set the `--calibration_size` to the desired number of images.

	```
	# python quantize.pyc \
	--deploy_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
	--train_val_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_train_val.prototxt \
	--weights /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel \
	--quantized_train_val_model q.train_val.prototxt \
	--calibration_directory ../../imagenet_val/ \
	--calibration_size 20
	```

6. The output used by XFDNN is the JSON file located in the same directory as the `--quantized_train_val_model argument`:
	```
	# ls -la ./q.train_val.json
	q.train_val.json
	```
   This file contains calibration parameters used by XFDNN through the XFDNN_QUANTIZE_CFGFILE as seen in the Caffe example scripts

[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[MxNet]:https://github.com/apache/incubator-mxnet
