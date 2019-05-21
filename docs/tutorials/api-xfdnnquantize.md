# xfDNN Quantizer

The xfDNN Quantizer performs a technique of quantization known as recalibration.

This technique does not require full retraining of the model, and can be accomplished in a matter of seconds, as you will see below. It also allows you to maintain the accuracy of the high precision model.

Quantization of the model does not alter the orginal high precision model, rather, it calculates the dynamic range of the model and produces scaling parameters recorded in a json file, which will be used by the xDNN overlay during execution of the network/model. Quantization of the model is an offline process that only needs to be performed once per model. The quantizer produces an optimal target quantization from a given network (Caffe - (prototxt and caffemodel) and Tensorflow (pb)) and calibration set (unlabeled input images) without requiring hours of retraining or a labeled dataset. The following sections describe the usage and command line arguments of the both `xfdnn_compiler_caffe`  and `xfdnn_compiler_tf` compiler.

To run the quantizer, use the command `python quantize.pyc`. The following sections describe the usage and commandline arguments of the quantizer.

## Usage

```cpp
quantize.pyc [-h]   [--framework FRAMEWORK] [--deploy_model DEPLOY_MODEL]
                    [--output_json OUTPUT_JSON] [--weights WEIGHTS]
                    [--calibration_directory CALIBRATION_DIRECTORY]
                    [--calibration_size CALIBRATION_SIZE]
                    [--calibration_seed CALIBRATION_SEED]
                    [--calibration_indices CALIBRATION_INDICES]
                    [--bitwidths BITWIDTHS] [--dims DIMS]
                    [--transpose TRANSPOSE] [--channel_swap CHANNEL_SWAP]
                    [--raw_scale RAW_SCALE] [--mean_value MEAN_VALUE]
                    [--input_scale INPUT_SCALE] [--input_layer INPUT_LAYER]
                    [--output_layer OUTPUT_LAYER]
                    
```

## Arguments

The table below describes the optional arguments.

Argument  |Description
--------- | ---------
-h, help  |Show this help message and exit
deploy_model DEPLOY_MODEL  |Input deploy prototxt file
output_json OUTPUT_JSON  |Output quantization file
weights WEIGHTS  |Input caffemodel file
calibration_directory CALIBRATION_DIRECTORY  |Directory containing calibration images
calibration_size CALIBRATION_SIZE  |Number of calibration images
calibration_seed CALIBRATION_SEED  |Seed with which to randomly sample calibration images, mutually exclusive with indices argument
calibration_indices CALIBRATION_INDICES  |Indices of sample calibration images, mutually exclusive with seed argument
bitwidths BITWIDTHS  |Bitwidths for input activations, parameters, and output activations
dims DIMS  |Image preprocessing operation (1 of 6) to crop/resize the input tensor while preserving the channel dimension
transpose TRANSPOSE  |Image preprocessing operation (2 of 6) to transpose dimensions, e.g., from H x W x K to K x H x W with "[2, 0, 1]"
channel_swap CHANNEL_SWAP  |Image preprocessing operation (3 of 6) to swap channels, e.g., from RGB to BGR with "[2, 1, 0]"
raw_scale RAW_SCALE  |Image preprocessing operation (4 of 6) to scale, e.g., from [0., 1.] to [0., 255.] with "255."
mean_value MEAN_VALUE  |Image preprocessing operation (5 of 6) to subtract from each channel, e.g., from BGR to B - 104., G - 117., and R - 123., respectively, with "[104., 117., 123.]"
input_scale INPUT_SCALE  |Image preprocessing operation (6 of 6) to scale, e.g., "1." for GoogLeNet and ResNet or "0.017" for MobileNet

## Additional Resources

* [Quantization Tutorial](https://github.com/Xilinx/ml-suite/blob/master/docs/tutorials/quantize.md)

* [Using the xfDNN Quantizer to quantize caffe models](https://github.com/Xilinx/ml-suite/blob/master/notebooks/quantizer_caffe.ipynb)

