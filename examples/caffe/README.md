## Getting Started to prepare and deploy a trained Caffe model for FPGA acceleration as an inference server 

### Running Caffe Benchmark Models
This directory provides scripts for running several well known models on the FPGA.
For published results, and details on running a full accuracy evaluation, please see these [instructions](Benchmark_README.md).

1. **One time setup**

   Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).

   ```
   cd /opt/ml-suite/examples/caffe
   python -m ck pull repo:ck-env
   python -m ck install package:imagenet-2012-val-min
   python -m ck install package:imagenet-2012-aux
   head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
   ```

   Resize all the images to a common dimension for Caffe

   ```
   python resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
   ```

   Get Samples Models

   ```
   python getModels.py
   ```

   Setup the Environment

   ```
   source $MLSUITE_ROOT/overlaybins/setup.sh
   ```

After the setup, run through a sample end to end caffe classification example using the following steps that demonstrates preparing and deploying a trained Caffe model for FPGA acceleration using Xilinx MLSuite**

  The following example uses the googlenet example. You can try the flow with also the other models found in /opt/models/caffe/ directory.

  ```
  cd /opt/ml-suite/examples/caffe 
  ``` 

2. **Prepare for inference**

  This performs quantization, compilation and subgraph cutting in a single step. To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.
  
  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler
 
  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.
 
  ```
   python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare
  ```

3. **Classification** - Run the validation set for 10 iterations on the FPGA

  ```
   python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --numBatches 10 --validate
  ```

4. **Classification** - Run a single image on the FPGA

  ```
  python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --image ../deployment_modes/dog.jpg
  ```

5. **Classification** - Running an Inference Server 

    Feel free to skip steps 5-7 from this readme and follow the instructions at [REST Server Example](examples/caffe/REST/README.md) to see an example of setting up an inference server, and running classifications as REST API to that server. 
 
    In this step, a python flask inference server is started, and the caffe model as well as the prototxt from the previous step are run on the FPGA to perform inference on an input image.

    This starts an inference server which can be accessed at port 5000 from end point /predict. For example: http://127.0.0.1:5000/predict

   ```
   python REST/app.py --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt --port 5000
   ```

   You can switch the above command to running in the background, and you can use a CURL command perform inference on an input image, passed to the REST inference server

6. **Classification** - Sending sample image requests to the Inference Server

   ```
   curl -X POST -F image=@$HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG 'http://localhost:5000/predict
   ```

   There is also a python script in the REST directory to do the same:

   ```
   python -m pip install requests --user
   python REST/request.py --rest_api_url http://localhost:5000/predict --image_path $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
   ```

7. **Benchmarking**

   Run the benchmark.py script in /opt/ml-suite/examples/caffe/ directory, which will send a sample input image to the inference server repeatedly while measuring response times, and finally calculating the average response time.

  ```
  python REST/benchmark.py --rest_api_url http://localhost:5000/predict --image_path /opt/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG


  The next section walks through how to invoke the individual steps namely quantize, compile and sub-graph cutting, that we already did as a single wrapper command in step 2. 

## Running the steps individually

1. **Quantize the model** - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  ```
   export DECENT_DEBUG=1 
  /opt/caffe/build/tools/decent_q quantize -model /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt -weights /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel -auto_test -test_iter 1 --calib_iter 1   
   ```
  
2. **Compile the Model** - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

  ```
  python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.pyc \
    -b 1 \
    -i 96 \
    -m 9 \
    -d 256 \
    -mix \
    --pipelineconvmaxpool \
    --usedeephi \
    --quant_cfgfile quantize_results/quantize_info.txt \
    -n quantize_results/deploy.prototxt \
    -w quantize_results/deploy.caffemodel \
    -g work/compiler \
    -qz work/quantizer \
    -C
  ```
   
3. **Subgraph Cutting** - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

   ```
   python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_subgraph.py \
    --inproto quantize_results/deploy.prototxt \
    --trainproto /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt \
    --outproto xfdnn_auto_cut_deploy.prototxt \
    --cutAfter data \
    --xclbin $MLSUITE_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin \
    --netcfg work/compiler.json \
    --quantizecfg work/quantizer.json \
    --weights work/deploy.caffemodel_data.h5 \
    --profile True
   ```


