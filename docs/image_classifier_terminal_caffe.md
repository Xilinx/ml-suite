## Running an image classifier with caffe model in docker containers.

1. Clone ML Suite
   
  ```
  git clone https://github.com/Xilinx/ml-suite.git
  ```
  
2. Download ml-suite container

  ```
  https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4.tar.gz
  ```

3. Load container

  ```
  sudo docker load < xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4.tar.gz
  ```
  
4. Run docker container

  ```
  $ cd ml-suite
  $ sudo ./docker_run.sh
  ```
  
5. One time setup

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
  
  Get necessary models
  ```
  cd /opt/ml-suite/examples/caffe
  python getModels.py
  ```
  
6. environment setup

  ```
  export MLSUITE_ROOT=/opt/ml-suite
  source $MLSUITE_ROOT/overlaybins/setup.sh
  ```

7. **Quantize the model** - The quantizer will generate scaling parameters for quantizing floats INT8. This is required, because FPGAs will take advantage of Fixed Point Precision, to achieve more parallelization at lower power

  ```
  cd /opt/ml-suite/share
  export DECENT_DEBUG=1
  /opt/caffe/build/tools/decent_q quantize -model /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt -weights /opt/models/caffe/bvlc_googlenet/bvlc_googlenet.caffemodel -auto_test -test_iter 1 --calib_iter 1
  ```

  This outputs quantize_info.txt, deploy.prototxt, deploy.caffemodel to $HOME/share/quantize_results/ directory

8. **Compile the Model** - In this step, the network Graph (prototxt) and the Weights (caffemodel) are compiled by the compiler
  
  ```
  python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.pyc -b 1 -i 96 -m 9 -d 256 -mix --pipelineconvmaxpool --usedeephi --quant_cfgfile quantize_results/quantize_info.txt -n quantize_results/deploy.prototxt -w quantize_results/deploy.caffemodel -g work/compiler -qz work/quantizer -C
  ```  
  
  This outputs compiler.json, quantizer.json and deploy.caffemodel_data.h5 to $HOME/share/work/ directory

9. **Subgraph Cutting** - In this step, the original graph is cut, and a custom FPGA accelerated python layer is inserted to be used for Inference.

  ```
  python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_subgraph.py --inproto quantize_results/deploy.prototxt --trainproto /opt/models/caffe/bvlc_googlenet/bvlc_googlenet_train_val.prototxt --outproto xfdnn_auto_cut_deploy.prototxt --cutAfter data --xclbin $MLSUITE_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin --netcfg work/compiler.json --quantizecfg work/quantizer.json --weights work/deploy.caffemodel_data.h5 --profile True
  ```  
  
  This output xfdnn_auto_cut_deploy.prototxt to $HOME/share/ directory
   
10. Running image classification for caffe model.

  ```
  python $MLSUITE_ROOT/examples/caffe/caffe_run.py --caffemodel $MLSUITE_ROOT/share/quantize_results/deploy.caffemodel --prototxt $MLSUITE_ROOT/share/xfdnn_auto_cut_deploy.prototxt --synset_words $MLSUITE_ROOT/examples/deployment_modes/synset_words.txt  --image $MLSUITE_ROOT/examples/deployment_modes/dog.jpg
  ```
