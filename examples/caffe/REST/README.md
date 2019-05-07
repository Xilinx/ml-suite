## Creating a REST API Inference server for Caffe Image Classification  

### One time setup if you haven't done these steps already

Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
The same dataset is used for mlperf inference benchmarks that are using imagenet.

```
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min
python -m ck install package:imagenet-2012-aux
head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt

# Resize all the images to a common dimension for Caffe
python $MLSUITE_ROOT/examples/caffe/resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256

# Get Samples Models 
python getModels.py

# Setup ml-suite Environment Variables
source $MLSUITE_ROOT/overlaybins/setup.sh

```

### Prepare for inference

This performs quantization, compilation and subgraph cutting in a single step. To run a Caffe model on the FPGA, it needs to be quantized, compiled, and a new graph needs to be generated. The new graph is similar to the original, with the FPGA subgraph removed, and replaced with a custom Python layer.

  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
   python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare
  ```

### Running an Inference Server for Classification

In this step, a python flask inference server is started, and the caffe model as well as the prototxt from the previous step are run on the FPGA to perform inference on an input image.

This starts an inference server which can be accessed at port 5000 from end point /predict. For example: http://127.0.0.1:5000/predict

   ```
   python app.py --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt --port 5000
   ```

You can switch the above command to running in the background, and you can use a CURL command perform inference on an input image, passed to the REST inference server

### Classification (Sending sample image requests to the Inference Server)

   ```
   curl -X POST -F image=@$HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG 'http://localhost:5000/predict
   ```

There is also a python script in the same directory to do the same:

   ```
   python -m pip install requests --user
   python request.py --rest_api_url http://localhost:5000/predict --image_path $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
   ```

### Running all of them in one step (preparing and serving up the inference server)

   ```
   ./run.sh
   ```

   ```
   !/bin/bash

   MODEL_WEIGHTS="/opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel"
   MODEL_PROTOTXT="/opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt"
   LABELS="/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt"
   PORT="5000"

   python ../run.py --prototxt ${MODEL_PROTOTXT} --caffemodel ${MODEL_WEIGHTS} --prepare

   python app.py --caffemodel ${MODEL_WEIGHTS} --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words ${LABELS} --port ${PORT}
   ```

### Benchmarking

   Run the benchmark.py script in /opt/ml-suite/examples/caffe/ directory, which will send a sample input image to the inference server repeatedly while measuring response times, and finally calculating the average response time.

  ```
  python benchmark.py --rest_api_url http://localhost:5000/predict --image_path /opt/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
  ```

