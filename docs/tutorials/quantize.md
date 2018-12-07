# xfDNN Quantizer

The xfDNN Quantizer performs a technique of quantization known as recalibration.

This technique does not require full retraining of the model, and can be accomplished in a matter of seconds, as you will see below. It also allows you to maintain the accuracy of the high precision model.

Quantization of the model does not alter the orginal high precision model, rather, it calculates the dynamic range of the model and produces scaling parameters recorded in a json file, which will be used by the xDNN overlay during execution of the network/model. Quantization of the model is an offline process that only needs to be performed once per model. The quantizer produces an optimal target quantization from a given network (prototxt and caffemodel) and calibration set (unlabeled input images) without requiring hours of retraining or a labeled dataset. The following sections describe the usage and command line arguments of the `xfdnn_compiler_caffe` compiler.

To run the quantizer, use the command `python quantize.pyc`. The following sections describe the usage and commandline arguments of the quantizer.

## Usage

```cpp
quantize.pyc [-h] [--deploy_model DEPLOY_MODEL]
                    [--output_json OUTPUT_JSON] [--weights WEIGHTS]
                    [--calibration_directory CALIBRATION_DIRECTORY]
                    [--calibration_size CALIBRATION_SIZE]
                    [--calibration_seed CALIBRATION_SEED]
                    [--calibration_indices CALIBRATION_INDICES]
                    [--bitwidths BITWIDTHS] [--dims DIMS]
                    [--transpose TRANSPOSE] [--channel_swap CHANNEL_SWAP]
                    [--raw_scale RAW_SCALE] [--mean_value MEAN_VALUE]
                    [--input_scale INPUT_SCALE]
```
		    
## Arguments

The table below describes the optional arguments.

Argument | Description
--------- | ---------
-h, help | Show this help message and exit
deploy_model DEPLOY_MODEL | Input deploy prototxt file
output_json OUTPUT_JSON | Output quantization file
weights WEIGHTS | Input caffemodel file
calibration_directory CALIBRATION_DIRECTORY | Directory containing calibration images
calibration_size CALIBRATION_SIZE | Number of calibration images
calibration_seed CALIBRATION_SEED | Seed with which to randomly sample calibration images, mutually exclusive with indices argument
calibration_indices CALIBRATION_INDICES | Indices of sample calibration images, mutually exclusive with seed argument
bitwidths BITWIDTHS | Bitwidths for input activations, parameters, and output activations
dims DIMS | Image preprocessing operation (1 of 6) to crop/resize the input tensor while preserving the channel dimension, e.g., to RGB color images of size 224x224 with "3, 224, 224"
transpose TRANSPOSE | Image preprocessing operation (2 of 6) to transpose dimensions, e.g., from H x W x C to C x H x W with "2, 0, 1"
channel_swap CHANNEL_SWAP | Image preprocessing operation (3 of 6) to swap channels, e.g., from RGB to BGR with "2, 1, 0"
raw_scale RAW_SCALE | Image preprocessing operation (4 of 6) to scale, e.g., from [0.0, 1.0] to [0.0, 255.0] with "255.0"
mean_value MEAN_VALUE | Image preprocessing operation (5 of 6) to subtract from each channel, e.g., from BGR to B - 104.0, G - 117.0, and R - 123.0, respectively, with "104.0, 117.0, 123.0"
input_scale INPUT_SCALE | Image preprocessing operation (6 of 6) to scale, e.g., "1.0" for GoogLeNet and ResNet or "0.017" for MobileNet

WARNING: Do not leave any argument unspecified as default values for unspecified arguments are subject to change.

## GoogLeNet v1 Example

1. Navigate to `$MLSUITE_ROOT/xfdnn/tools/quantize`
	```
	# cd $MLSUITE_ROOT/xfdnn/tools/quantize
	```

2. This next command will execute GoogLeNet-v1 quantization using the deploy models provided in the $MLSUITE_ROOT/models directory. This quantization expects at least eight images to be available in the `$MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal` directory.  Refer to http://www.image-net.org/download-imageurls for downloading ILSVRC files from ImageNet. Other files may be used which do not require any special file naming convention.
	```
	# python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.pyc \
        --deploy_model $MLSUITE_ROOT/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_int8_deploy.json \
        --weights $MLSUITE_ROOT/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel \
        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
        --calibration_size 32 \
        --bitwidths 8,8,8 \
        --dims 3,224,224 \
        --transpose 2,0,1 \
        --channel_swap 2,1,0 \
        --raw_scale 255.0 \
        --mean_value 104.0,117.0,123.0 \
        --input_scale 1.0
	```

	Instead of randomly sampling 32 images without replacement from the calibration directory, set the `--calibration_indices` to provide a specific sample of images from the calibration directory (based on an ascending alphanumerical sorting of calibration directory).

	```
	# python $MLSUITE_ROOT/xfdnn/tools/quantize/quantize.pyc \
        --deploy_model $MLSUITE_ROOT/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn_deploy.prototxt \
        --output_json $MLSUITE_ROOT/examples/quantize/work/caffe/bvlc_googlenet_without_lrn/bvlc_googlenet_without_lrn_quantized_int8_deploy.json \
        --weights $MLSUITE_ROOT/models/caffe/bvlc_googlenet_without_lrn/fp32/bvlc_googlenet_without_lrn.caffemodel \
        --calibration_directory $MLSUITE_ROOT/models/data/ilsvrc12/ilsvrc12_img_cal \
        --calibration_size 32 \
        --calibration_indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31 \
        --bitwidths 8,8,8 \
        --dims 3,224,224 \
        --transpose 2,0,1 \
        --channel_swap 2,1,0 \
        --raw_scale 255.0 \
        --mean_value 104.0,117.0,123.0 \
        --input_scale 1.0
	```

6. The output used by XFDNN is the JSON file as specified by `--output_json OUTPUT_JSON`:
	```
	# ls $MLSUITE_ROOT/examples/quantize/work/caffe/bvlc_googlenet_without_lrn
	bvlc_googlenet_without_lrn_quantized_int8_deploy.json
	```
   This file contains calibration parameters used by XFDNN through `--quantcfg QUANTCFG`, as seen in the Caffe example scripts.

[here]: launching_instance.md
[click here]: https://github.com/aws/aws-fpga/blob/master/sdk/userspace/fpga_mgmt_tools/README.md#sudo-or-root-privileges
[MxNet]:https://github.com/apache/incubator-mxnet

## Additional Resources

* [Quantization Tutorial](https://github.com/Xilinx/ml-suite/blob/master/docs/tutorials/quantize.md)

* [Using the xfDNN Quantizer to quantize caffe models](https://github.com/Xilinx/ml-suite/blob/master/notebooks/quantizer_caffe.ipynb)
