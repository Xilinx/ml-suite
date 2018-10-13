# xfDNN Compiler 

The Xilinx Machine Learning (ML) Suite Compiler provides users with the tools to develop and deploy Machine Learning applications for real-time inference.

The compiler script interfaces with ML Frameworks such as Caffe and Tensorflow to read deep learning networks, and then generates a sequence of instructions for the xfDNN framework to execute on the FPGA. This includes a computational graph traversal, node merging and optimization, memory allocation and, finally, instruction generation.

For instructions on launching and connecting to aws instances, see [here](https://github.com/Xilinx/ml-suite/blob/master/docs/tutorials/docs/tutorials/launching_instance.md).

Each ML Framework has a different version of the compiler in the ml-suite/xfdnn/tools/compile/bin/ directory:

* Caffe - `xfdnn_compiler_caffe.pyc`

* Keras - `xfdnn_compiler_keras.pyc`

* MxNet - `xfdnn_compiler_mxnet.pyc`

* Tensforflow - `xfdnn_compiler_tensorflow.pyc`

Each of these tools contain largely the same arguments. The following sections describe the usage and command line arguments of the `xfdnn_compiler_caffe` compiler.

## Usage

```cpp
xfdnn_compiler_caffe.pyc [-h] [-n NETWORKFILE] [-g GENERATEFILE]
                                [-w WEIGHTS] [-o PNGFILE] [-c CONCATSTRATEGY]
                                [-s STRATEGY] [--schedulefile SCHEDULEFILE]
                                [-i DSP] [-v] [-a ANEW] [-m MEMORY] [-d DDR]
                                [-p PHASE]
```

## Arguments

The table below describes the optional arguments.

Argument  |Description
--------- | ---------
-h, help  |Show this help message and exit
-n NETWORKFILE, networkfile NETWORKFILE  |Main prototxt for compilation
-g GENERATEFILE, generatefile GENERATEFILE  |Output file instructions
-w WEIGHTS, weights WEIGHTS  |Input caffemodel file to generate weight for python
-o PNGFILE, pngfile PNGFILE  |Write Graph in PNG file format. Requires dot executable
-c CONCATSTRATEGY, concatstrategy CONCATSTRATEGY  |
-s STRATEGY, strategy STRATEGY  |
schedulefile SCHEDULEFILE  |
-i DSP, dsp DSP  |
-v, verbose  |
-a ANEW, anew ANEW  |prefix of the new prototext
-m MEMORY, memory MEMORY  |
-d DDR, ddr DDR  |
-p PHASE, phase PHASE  |

## Additional Resources

* [xfdnn Compiler Tutorial](https://github.com/Xilinx/ml-suite/blob/master/docs/tutorials/compile.md)

* [Using the xfDNN Compiler w/ a Caffe Model](https://github.com/Xilinx/ml-suite/blob/master/notebooks/compiler_caffe.ipynb)


