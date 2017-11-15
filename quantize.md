# Quantization Tutorial

## Introduction
The quantize script produces an optimal target quantization within a matter of minutes from a given network (prototxt and caffemodel) and calibration set (unlabeled input images) without requiring hours of retraining or a labeled dataset.

For instructions on launching and connecting to instances, see [here][].


## Support Features

To run the quantizer, use the command `python quantize.pyc`

List of Arugments available:

- `[-h]`
- `[--deploy_model DEPLOY_MODEL]` - Input prototxt for calibration
- `[--train_val_model TRAIN_VAL_MODEL]` - Input training prototxt for calibration
- `[--weights WEIGHTS]` - FP32 pretrained caffe model
- `[--quantized_deploy_model QUANTIZED_DEPLOY_MODEL]` - Calibrated output prototxt
- `[--quantized_train_val_model QUANTIZED_TRAIN_VAL_MODEL]` - Calibrated training prototxt for calibration
- `[--quantized_weights QUANTIZED_WEIGHTS]` Calibrated output caffe model
- `[--calibration_directory CALIBRATION_DIRECTORY]` - Dir of dataset of original images
- `[--calibration_size CALIBRATION_SIZE]` - Number of images to use for calibration, default is 8
- `[--calibration_indices CALIBRATION_INDICES]`
- `[--bitwidths BITWIDTHS] [--dims DIMS]`
- `[--transpose TRANSPOSE] [--channel_swap CHANNEL_SWAP]`
- `[--raw_scale RAW_SCALE] [--mean_value MEAN_VALUE]`

Any argument that is not passed will be set with a default value.
Default values are located here: `xfdnn_17_11_13/models/bvlc_googlenet_without_lrn/fp32`

## GoogLeNet v1 Example

1. Connect to F1
2. Navigate to `/xfdnn_17_11_13/frameworks/`
	```
	$ ls
	caffe_docker  docker_build_caffe.sh  docker_run_caffe.sh  xlnx_docker
	```

3. Execute `./docker_run_caffe.sh` to enter application docker
	```
	$ ./docker_run_caffe.sh
	/opt$
	```
4. Navigate to `/xlnx/xfdnn_tools/quantize/`
	```
	$ cd /xlnx/xfdnn_tools/quantize/
	```

5. Execute the quantize script and pass a directory of images to calibrate on. For this example, we are using all the defaults of GoogLeNet.
	```
	$ python quantize.pyc --calibration_directory ../../imagenet_val/
	```
	Once the script is complete, the console will display the full quatize parameters.

	```
	quantize.pyc --deploy_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt --train_val_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_train_val.prototxt --weights /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel --quantized_deploy_model /xlnx/models/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_deploy.prototxt --quantized_train_val_model /xlnx/models/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_train_val.prototxt --quantized_weights /xlnx/models/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized.caffemodel --calibration_directory ../../imagenet_val/ --calibration_size 8 --calibration_indices 7720,7825,13145,13963,27567,28835,35295,46345 --bitwidths 8,8,8 --dims 3,224,224 --transpose 2,0,1 --channel_swap 2,1,0 --raw_scale 255.0 --mean_value 104,117,123 --input_scale 1.0
	```

	If only a calibration_directory is provided, the script will utilize default parameters, which are set to GoogLeNet v1, and will use 8 images from the calibration_directory.

	To run on more than 8 images (recommended is atleast 20), set the `--calibration_size` to the desired number.
	```
	$ python quantize.pyc --calibration_directory ../../imagenet_val/ --calibration_size 20
	[output truncated]
	quantize.pyc --deploy_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt --train_val_model /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_train_val.prototxt --weights /xlnx/models/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel --quantized_deploy_model /xlnx/models/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_deploy.prototxt --quantized_train_val_model /xlnx/models/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_train_val.prototxt --quantized_weights /xlnx/models/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized.caffemodel --calibration_directory ../../imagenet_val/ --calibration_size 20 --calibration_indices 963,2945,6586,8120,9584,14051,14063,14915,21814,26904,27202,29626,30109,38218,38750,41186,45414,46803,48362,48412 --bitwidths 8,8,8 --dims 3,224,224 --transpose 2,0,1 --channel_swap 2,1,0 --raw_scale 255.0 --mean_value 104,117,123 --input_scale 1.0
	```

[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[MxNet]:https://github.com/apache/incubator-mxnet
