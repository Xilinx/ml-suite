# Compiler

YACC from CNN to pseudo FPGA code  

## Software Overview:

The Xilinx Machine Learning (ML) Suite Compiler provides users with the tools to develop and deploy Machine Learning (ML) applications for Real-time Inference. It provides support for Caffe, MxNet, Keras, and Tensorflow (TF) as well as Python.

Following is the structure of the compiler as modules:

```
.
|-- README.md
|-- __init__.py
|-- bin
|   |-- __init__.py
|   |-- messages.py
|   |-- modules.py
|   |-- xfdnn_compiler_base.py
|   |-- xfdnn_compiler_caffe.py
|   |-- xfdnn_compiler_graffite.py
|   |-- xfdnn_compiler_keras.py
|   |-- xfdnn_compiler_mxnet.py
|   `-- xfdnn_compiler_tensorflow.py
|-- codegeneration
|   |-- backendtools.py
|   |-- hardware.py
|   `-- hardwarecodegeneration.py
|-- fpga_definition
|   |-- fpga_code
|   `-- fpga_graph
|-- frontends
|   |-- __init__.py
|   `-- frontend_caffe.py
|-- graph
|   |-- dagtools.py
|   |-- dagtools_replication.py
|   `-- dagtools_type.py
|-- memory
|   |-- memorytools.py
|   |-- memorytools_code_splitting.py
|   |-- memorytools_code_splitting_v3.py
|   `-- memorytools_twolevel.py
|-- network
|   |-- __init__.py
|   |-- graffite_tools.py
|   |-- keras_tools.py
|   |-- medium-load.py
|   |-- medium-tffreeze-1.py
|   |-- mxnetjson.py
|   |-- node_info.py
|   |-- tensor_tools.py
|   |-- tools.py
|   `-- tools_type.py
|-- optimizations
|   |-- graphoptimization.py
|-- parallel
|   `-- parallelism.py
`-- weights
    `-- weights.py
```

The following section describes the compiler flow. Although the code contains comments and spelling errors, it serves to
 inform the authors and helps the reader navigate what is a lot of information. You can access most of the code comments using
the python help() command. 


## Network: Front end and data Structure   

Every compiler starts with a front end. We have five preliminary
front ends, the main ones being Caffe and TensorFlow. A
PyDot graph is created which represents a computation layer and a
tensor/blob. Each tensor has an associated operation that
generates the tensor once. A DAG without re-computation and for
inference.


* **'tools_type.py'**: This module represents the backbone of our
  intermediate representation.

* **'tools.py'**: This module is the Caffe front end. Caffe
  creates two networks; one is static and contains information such as
  the strides for convolution, the other is dynamic and contains weights for a convolution. We summarize the network into
  one and we prefer information coming from the static. We take only deploy networks.

* **'medium-load.py'**, **'medium-tffreeze-1.py'**, and **'tensor_tools.py'**: These are
  our attempts to reign in TF networks. TF philosophy aims to have a
  single network describing both training and inference. Because Inference
  should be streamlined, there are a few things that should be done
  in between freezing the models, prior to compilation.

* **'graffite_tools.py'**: An internal format used to compile code for
  a math engine.

* **'keras_tools.py'**: Because Keras is based on TF, it may be tempting to use TF directly. However, Keras uses TF utilities
  so that inference models are based on conditional branches that
  neither the compiler nor the final FPGA code can handle well. This
  front end is designed to skip the intermediate TF representation and capture
  the Keras network. This has not been touched recently.

* **'mxnetjson.py'**: MxNET is the simplest to understand and use. It consists of a graph, which is the static description
 of the computation, and a dictionary, which contains the weights of the model.


Relative to Caffe, TF, and Graffite, Keras and MxNet are less commonly used and less developed.



## Graph: Representation and Manipulation

PyDot is used for visualization, but because of a poor interface for collecting neighbors' information, custom modifications are applied to the graph and its information. Graph optimizations
become independent of the original framework and therefore can be tailored to FPGAs. The main goal of this phase is to compute memory requirements (that is, how much URAM is required for the input manipulation model, weights excluded).


There are three modules: 

* **'dagtools_type.py'**: Describes further data structures that are added
  to the ParameterLayer. Consult the code for more information. For graph algorithms, available information includes: color
  information, schedule information to express when a layer is
  executed, and information relating to when a tensor is computed and used.

* **'dagtools_type.py'**: In combination with a schedule, this is the graph manipulation "work-horse". If the graph is changed (for example, a layer is added) it needs to be added into the schedule in accordance with a policy.

* **'dagtools_replication.py'**: For V3 architecture, a tensor has shapes
  and space optimizations that can be applied as soon as
  the graph is initialized. Replication is a memory layout dedicated to improve
  throughput of the computation.


In the past, graph manipulation did not change the weights of
the layers (graph optimization was relatively trivial). Through time (as
more front ends have been introduced), the philosophy has changed and now graph manipulations and graph optimizations should be considered as part of the same phase.

## Memory: Allocation, De-allocation, DDR, and Computation from DDR

Memory management is necessary when working with large
images. This module deals with both "problem" sized images that fit into an arbitrary bounded Activation Memory (AM - a
fast URAM) and those that don't, which use DDR.

Concat layers are a composition of multiple tensors
which are handled in memory as a single tensor. The are allocated as
single tensor plus offsets and they are moved across levels as single
tensors to avoid inconsistencies.

* **'memorytools.py'**: The main module, used for memory
  allocation with and without DDR (that is, LRU Cache). 

* **'memorytools_twolevel.py'**: The module used for two level
  memory (DDR + AM).

* **'memorytools_code_splitting.py'**: The data must reside in DDR and
  the computation is split into tiles by software Gather,
  Compute, Scatter ( multiple instructions).

* **'memorytools_code_splitting_v3.py'** The data must reside in DDR
  and the computation is split into tiles but the compiler suggests
  only the maximum tile size (a single instruction).

## Graph Optimization

Graph optimization means that not only can graphs change, but also the weights of a layer. For example in caffe,
some graph manipulation such as telescoping of BN into convolution does
not change the weights. However, most graph manipulations are
becoming graph optimizations. Typically in TF, graph optimizations are deployed.

## Code Generation

As function of the FPGA and IP available, codes or instructions need to be created. Following is the legacy code for up to V2
`backendtools.py` and the new way to introduce hardware abstractions:

* **'hardware.py'**: You can create different architectures and abstract
  their behaviors. For example, volumes in DDR have different space
  constraints than in AM. DSP kernels allow replication of
  volumes. Some require the volumes to be aligned by channels, and some
  by width.

* **'hardwarecodegeneration.py'**: Creates a json output, a cmd file, and a
  cleanout file (from the graph).

Hardware abstraction is used throughtout the compilation, not only in code generation.

## Weights

After creating the instructions, the weights and
biases of the layers need to be stored. Follow the schedule and store them as TXT
files. These will be read at run time and stored by the system.

The weights of the FC (inner products) that are not executed by the FPGA are also stored. These are executed by the runtime environment.

## Parallel

Most of the above work is done when two inputs are given; Graph and Schedule.

This module descibes how to look for different schedules. In
particular, those that exploit parallelism between
pool and convolutions, exploit a smaller memory foot print, or have
a different topological order. This module is the first attempt to
introduce compiler optimizations that are not ruled-based, and can
touch different parts of the compilation.

## Compilers

The 'bin' directory contains examples of compilers. Each compiler
may have a different flow because of how each framework addresses the
same network. TF and Caffe networks may use different strategies and schedules. We can therefore customize the compiler
for a specific environment. At the same time we can emphasize the
common functionality into a single class: 'xfdnn_compiler_base.py'


```
parameters = [
    ("-g", "--generatefile",    str,None,     'store',"Output file instructions"),
    ("-o", "--pngfile",         str,None,     'store',"Write Graph in PNG file, Requires dot executable"),
    ("-c", "--concatstrategy",  str,None,     'store',None),
    ("-s", "--strategy",        str,"all",    'store',"Heuristics for memory allocation: [all] + "+str(hardware.STRATEGIES)),
    ("-j", "--versionjson",     str,None,     'store',argparse.SUPPRESS),
    ("-L", "--lasttensorbyname",str,None,     'store',"Return the size of the last tensors by name"),
    (None, "--schedulefile",    str,None,     'store',"Give information about schedule and memory allocation "),
    ("-i", "--dsp",             int,None,     'store',"DSP size [common 28,56 V2 ,96 V3"),
    ("-b", "--bytesperpixels",  int,2,        'store',None),
    ("-v", "--verbose",         bool,False,   'store_true',"WARNING: This will be SO MUCH SO VERY VERBOSE" ),
    ("-pcmp", "--pipelineconvmaxpool",         bool,False, 'store_true',"Activate the conv+maxpool pipeline"),
    ("-P", "--parallelism",     bool,False,   'store_true',"Full search for parallel conv+pool"),
    ("-N", "--noreplication",   bool,False,   'store_true',"No replication for V3"),
    ("-R", "--barrier",         bool,False,   'store_true',argparse.SUPPRESS),
    ("-x", "--approximate",     bool,False,   'store_true',"Approximation for the collection of max_items (bottles)"),
    ("-m", "--memory",          float,None,     'store',"ACtivation Memory size in MB"),
    ("-d", "--ddr",             int,256,     'store',None,"DDR Allocated to kernel in MB"),
    ("-M", "--manasadebugmode",       bool,False,   'store_true',argparse.SUPPRESS),
    ("-t", "--fromtensorflow",  bool,False,   'store_true',"This is a Caffe network translted from TF (SSA)"),
    ("-C", "--cpulayermustgo",  bool,False,   'store_true',"WARNING: Please keep it False"),
    ("-2", "--conv_1x1_s2",     bool,False,   'store_true',"This is used for V2 and resnet: conv -> maxpool+ conv "),
    ("-4", "--poolingaround",     bool,False,   'store_true',argparse.SUPPRESS),
    ("-3", "--dedicateddsp",    str,None,    'store',"Only V3 small or big block first layer"),
    ("-B", "--bridges",     bool,False,   'store_true',"It introduces identity scales so we can work out Concat of Concat"),
    ("-G", "--godreplication",  str,None,     'store',"Hardware people can set their replicaiton: only V3 and only V3's Gods"),
    ("-r", "--rankdir",         str,"BT",     'store',argparse.SUPPRESS)]
```

Please review the `BaseInitialization` function to see how this parameter sets the compiler and its execution.

Caffe is still the dominant framework. TF is a far second. Keras and MxNet have not been developed for some time. However, as you browse the code, the compiler is more and more standard in the sense that the same major flow is used and shared in the `_base`.

[Slides](https://gitenterprise.xilinx.com/acceleration/MLsuite/blob/master/xfdnn/tools/compile/bin/XFDNN%20Compiler%20in%20a%20nutshell.pdf)
