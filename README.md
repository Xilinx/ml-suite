<table style="width:100%">
<tr>
<th width="100%" colspan="6"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Xilinx ML Suite v1.4</h2>
</th>
</table>
  

The Xilinx Machine Learning (ML) Suite provides users with the tools to develop and deploy Machine Learning applications for Real-time Inference. It provides support for many common machine learning frameworks such as Caffe, Tensorflow, and MXNet.  

![](docs/img/stack.png)

The ML Suite is composed of three basic parts:
1. **xDNN IP** - High Performance general CNN processing engine.
2. **xfDNN Middleware** - Software Library and Tools to Interface with ML Frameworks and optimize them for Real-time Inference.
3. **ML Framework and Open Source Support**  - Support for high level ML Frameworks and other open source projects.

**Learn More:** [ML Suite Overview][]  
**Watch:** [Webinar on Xilinx FPGA Accelerated Inference][]   
**Forum:** [ML Suite Forum][]

## [See What's New](docs/release-notes/1.x.md)
 - [Release Notes][]
 - Integration of Deephi DECENT Quantizer for Caffe
 - xfDNN Runtime API upgraded to support multi-output networks
 - Ease of use enhancements
    - Docker Images
    - Run on FPGA using Caffe's custom Python layer
 
## Getting Started with Docker Container and Jupyter Notebook

1. Pull the MLsuite docker container 
   ```
   docker pull <path-to-docker-image-on-dockerhub>
   docker load <path-to-docker-image>
   ``` 

2. Launch the MLSuite Container

   ```
   docker run \
   --rm \
   --net=host \
   --privileged=true \
   -it \
   -v /dev:/dev \
   -v /opt/xilinx:/opt/xilinx \
   -w /opt/ml-suite \
   xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
   bash
   ```

3. Inside the container, Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
   

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

4. Setup the Environment 
   
   ```
   source $MLSUITE_ROOT/overlaybins/setup.sh
   ```

5. Start the Interactive Caffe Image Classification Jupyter Notebook 
   
   The docker container has an example jupyter notebook which demonstrates the steps required to prepare and deploy a trained Caffe model for FPGA acceleration using Xilinx MLSuite:

    **Quantize the model** - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power.
   
    **Compile the Model** - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

    **Subgraph Cutting** - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.
   
    **Classification** - In this step, the caffe model and the prototxt from the previous step are run on the FPGA to perform inference on an input image.
 
  
   ```
   cd /opt/ml-suite/notebooks
   jupyter notebook --no-browser --ip=0.0.0.0
   ```
   
   Open the Jupyter Notebook on the browser and go through the steps interactively 


## Getting Started with Docker Container and Command Line to prepare and deploy a trained Caffe model for FPGA acceleration as an inference server 

1. Pull the MLsuite docker container
   
   ```
   docker pull <path-to-docker-image-on-dockerhub>
   docker load <path-to-docker-image>
   ```

2. Launch the MLSuite Container

   ```
   docker run \
   --rm \
   --net=host \
   --privileged=true \
   -it \
   -v /dev:/dev \
   -v /opt/xilinx:/opt/xilinx \
   -w /opt/ml-suite \
   xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
   bash
   ```

3. Inside the container, Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).

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

4. Setup the Environment

   ```
   source $MLSUITE_ROOT/overlaybins/setup.sh
   ```

5. Run through a sample end to end caffe classification example that demonstrates the steps required to prepare and deploy a trained Caffe model for FPGA acceleration using Xilinx MLSuite**

  The following example uses the googlenet example. You can try the flow with also the other models found in /opt/models/caffe/ directory.

  ```
  cd /opt/ml-suite/examples/caffe 
  ``` 

6. **Prepare for inference**

  This performs quantization, compilation and subgraph cutting in a single step.
  
  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler
 
  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.
 
  ```
   python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare
  ```

7. Run the validation set for 10 iterations on the FPGA

  ```
   python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --numBatches 10 --validate
  ```

8. Run a single image on the FPGA

  ```
  python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --image ../deployment_modes/dog.jpg
  ```

  You can skip to step-12 if you want to see how to set up an inference server serving a REST API. The next few steps 9, 10, 11 walk through how to invoke the individual steps namely quantize, compile and sub-graph cutting, that we already did as a single wrapper command in step 6. 

9. **Quantize the model** - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  ```
   export DECENT_DEBUG=1 
  /opt/caffe/build/tools/decent_q quantize -model /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt -weights /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel -auto_test -test_iter 1 --calib_iter 1   
   ```
  
10. **Compile the Model** - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

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
   
11. **Subgraph Cutting** - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

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

12. **Running an Inference Server for Classification** - In this step, a python flask inference server is started, and the caffe model as well as the prototxt from the previous step are run on the FPGA to perform inference on an input image.

    This starts an inference server which can be accessed at port 5000 from end point /predict. For example: http://127.0.0.1:5000/predict

   ```
   python app.py --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt --port 5000
   ```  

   You can switch the above command to running in the background, and you can use a CURL command perform inference on an input image, passed to the REST inference server

13. **Classification (Sending sample image requests to the Inference Server)**

   ```
   curl -X POST -F image=@$HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG 'http://localhost:5000/predict
   ```

   There is also a python script in the same directory to do the same:

   ```
   python -m pip install requests --user
   python request.py --rest_api_url http://localhost:5000/predict --image_path $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
   ```
14. **Benchmarking**
    
   Run the benchmark.py script in /opt/ml-suite/examples/caffe/ directory, which will send a sample input image to the inference server repeatedly while measuring response times, and finally calculating the average response time.

  ```
  python benchmark.py --rest_api_url http://localhost:5000/predict --image_path /opt/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
  ```

## Running a pipeline of docker containers to run the various steps to prepare and deploy a caffe model, and serve with an inference server

1. Create a shared directory to map with the container for sharing intermediate results between various stages

  ```
  mkdir $HOME/share
  sudo chmod -R 777 $HOME/share
  ```

2. Clone ML Suite
   
  ```
  git clone https://github.com/Xilinx/ml-suite.git
  ```

3. Download a minimal validation set for [Imagenet2012](http://www.image-net.org/challenges/LSVRC/2012) using [Collective Knowledge (CK)](https://github.com/ctuning).
The same dataset is used for mlperf inference benchmarks that are using imagenet.

  You can install CK from PyPi via pip install ck or pip install ck --user.

  You can then test that it works from the command line via ck version or from your python environment as follows:

  ```
  $ python

  > import ck.kernel as ck
  > ck.version({})
  ```
  
  Refer to https://github.com/ctuning/ck for install instructions

  ```
  python -m ck pull repo:ck-env 
  python -m ck install package:imagenet-2012-val-min
  python -m ck install package:imagenet-2012-aux
  head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
  ```

  Resize all the images to a common dimension for Caffe
  ```
  python -m pip --no-cache-dir install opencv-python --user 
  python MLsuite/examples/caffe/resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
  ``` 

  Move $HOME/CK-TOOLS to $HOME/share/CK-TOOLS
  ```
  mv $HOME/CK-TOOLS $HOME/share/CK-TOOLS
  ```
  
3. Pull the MLsuite docker container
  
  ```
  docker pull <path-to-docker-image-on-dockerhub>
  docker load <path-to-docker-image>
  ```

In the next few steps, we will launch individual instances of the container to run the quantizer, compiler, sub-graph cutter and inference steps of the image classification. This  will allow us to horizontally as well as vertically scale the relevant steps.

While invoking docker, make sure to mount $HOME/share and $HOME/share/CK-TOOLS directories to the appropriate locations within the container as shown in the various steps below.
   
4. **Prepare for inference**

  This performs quantization, compilation and subgraph cutting in a single step.

  Quantize the model - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  Compile the Model - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler

  Subgraph Cutting - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
   docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/share/CK-TOOLS:/home/mluser/CK-TOOLS \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python run.py --prototxt /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prepare
  ```

  You can skip to step-8 if you want to see how to set up an inference server serving a REST API. The next few steps 5, 6, 7 walk through how to invoke the individual steps namely quantize, compile and sub-graph cutting, that we already did as a single wrapper command in step 4.

5. **Quantize the model** - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/share/CK-TOOLS:/home/mluser/CK-TOOLS \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && export DECENT_DEBUG=1 && /opt/caffe/build/tools/decent_q quantize -model /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt -weights /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel -auto_test -test_iter 1 --calib_iter 1'
  ```

  This outputs quantize_info.txt, deploy.prototxt, deploy.caffemodel to $HOME/share/quantize_results/ directory

6. **Compile the Model** - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled, the compiler
  
  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/CK-TOOLS:/home/mluser/CK-TOOLS \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.pyc -b 1 -i 96 -m 9 -d 256 -mix --pipelineconvmaxpool --usedeephi --quant_cfgfile quantize_results/quantize_info.txt -n quantize_results/deploy.prototxt -w quantize_results/deploy.caffemodel -g work/compiler -qz work/quantizer -C' 
  ```  

7. **Subgraph Cutting** - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -a stdin -a stdout -a stderr \
  -t \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/CK-TOOLS:/home/mluser/CK-TOOLS \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_subgraph.py --inproto quantize_results/deploy.prototxt --trainproto /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --outproto xfdnn_auto_cut_deploy.prototxt --cutAfter data --xclbin $MLSUITE_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin --netcfg work/compiler.json --quantizecfg work/quantizer.json --weights work/deploy.caffemodel_data.h5 --profile True'
  ```      
   
8. **Inference REST API Server for Classification** - In this step, a python flask inference server is started, which takes in the caffe model as well as the prototxt from the previous step to run on the FPGA, and exposes a REST API endpoint to perform inference on input images

   This starts an inference server which can be accessed at port 5000 from end point /predict. For example: http://127.0.0.1:5000/predict

  ```
  docker run \
  --rm \
  --net=host \
  --privileged=true \
  -d \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -w /opt/ml-suite \
  -v $HOME/share:/opt/share \
  -v $HOME/CK-TOOLS:/home/mluser/CK-TOOLS \
  xilinxatg/ml_suite:ubuntu-16.04-caffe-mls-1.4 \
  bash -c 'source /opt/ml-suite/overlaybins/setup.sh && cd /opt/share/ && python /opt/ml-suite/examples/caffe/app.py --caffemodel /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel --prototxt xfdnn_auto_cut_deploy.prototxt --synset_words /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/synset_words.txt --port 5000'
  ```
   
9. **Classification (Passing an Image for Inference)** - CURL command to test the inference server REST API end point by passing images to perform inference

   ```
   curl -X POST -F image=@$HOME/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG 'http://localhost:5000/predict'
   ```

10. **Benchmarking**

  ```
  python
  ```
  In the python shell, copy paste the following python script which will send a few image requests to the inference server running from step 8 and get back predictions. This script will send a sample input image to the server repeatedly while measuring response times, and finally calculating the average response time.

  ```
  # import the necessary packages
  import os
  import requests

  HOME = os.getenv("HOME")

  # initialize the REST API endpoint URL along with the input
  # image path
  SERVER_URL = "http://localhost:5000/predict"
  IMAGE_PATH = HOME+"/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG"

  # load the input image and construct the payload for the request
  image = open(IMAGE_PATH, "rb").read()
  payload = {"image": image}

  # submit the request

  total_time = 0
  num_requests = 10
  for _ in xrange(num_requests):
    response = requests.post(SERVER_URL, files=payload)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()

  print(' avg latency: {} ms'.format((total_time*1000)/num_requests))
  ```

  ```
  exit()
  ```    

  You can run the benchmark.py script in /opt/ml-suite/examples/caffe/ directory as follows. This script will send a sample input image to the inference server repeatedly while measuring response times, and finally calculating the average response time.
 

  ```
  python benchmark.py --rest_api_url http://localhost:5000/predict --image_path /opt/share/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG
  ``` 

## Downloading Overlays and Pre-Trained Models from [ML Suite Lounge][]
   
   The [ML Suite Lounge][] contains up-to-date overlays as well as pre-trained models. You can download them for using in your containers or workflow at any point.
  
   - Overlays: Download and unzip desired overlays into the `ml-suite/overlaybins/` dir in the container, for example: `ml-suite/overlaybins/alveo-u200`
   - Pre-Trained Models: Download and unzip to the `/ml-suite/` dir in the container. 
  
## References 
- [ML Suite Overview][]  
- Tutorials and Examples:
  - [Jupyter Notebooks](notebooks/)
  - [Precompiled Deployment Examples](examples/deployment_modes/)  

## Recommended System Requirements
- OS: Ubuntu 16.04.2 LTS, Ubuntu 18.04 LTS, CentOS 7.4
- CPU: 6 Cores (Intel/AMD/Power9)
- Memory: 8 GB

## Supported Platforms
Cloud Services
 - [Amazon AWS EC2 F1][]
 - [Nimbix](https://www.nimbix.net/xilinx/)

 On Premise Platforms (Visit [ML Suite Lounge] for Details)
  - Alveo U200 Data Center Accelerator Card
  - Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit

## Questions and Support

- [FAQ][]
- [AWS F1 Application Execution on Xilinx Virtex UltraScale Devices][]
- [ML Suite Forum][]
- [Performance Whitepaper][]

[models]: docs/models.md
[Amazon AWS EC2 F1]: https://aws.amazon.com/marketplace/pp/B077FM2JNS
[Xilinx Virtex UltraScale+ FPGA VCU1525 Acceleration Development Kit]: https://www.xilinx.com/products/boards-and-kits/vcu1525-a.html
[AWS F1 Application Execution on Xilinx Virtex UltraScale Devices]: https://github.com/aws/aws-fpga/blob/master/SDAccel/README.md
[SDAccel Forums]: https://forums.xilinx.com/t5/SDAccel/bd-p/SDx
[Release Notes]: docs/release-notes/1.x.md
[UG1023]: https://www.xilinx.com/support/documentation/sw_manuals/xilinx2017_4/ug1023-sdaccel-user-guide.pdf
[FAQ]: docs/faq.md
[ML Suite Overview]: docs/ml-suite-overview.md
[Webinar on Xilinx FPGA Accelerated Inference]: https://event.on24.com/wcc/r/1625401/2D3B69878E21E0A3DA63B4CDB5531C23?partnerref=Mlsuite
[ML Suite Forum]: https://forums.xilinx.com/t5/Xilinx-ML-Suite/bd-p/ML 
[ML Suite Lounge]: https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html
[Models]: https://www.xilinx.com/products/boards-and-kits/alveo/applications/xilinx-machine-learning-suite.html#gettingStartedCloud
[whitepaper here]: https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf
[Performance Whitepaper]: https://www.xilinx.com/support/documentation/white_papers/wp504-accel-dnns.pdf
