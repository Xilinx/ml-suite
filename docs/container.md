## Getting Started with the ML Suite Docker Container
The ML Suite Docker container provides users with a largely self contained software environment for running inference on Xilinx FPGAs.
The only external dependencies are:  
- docker-ce
- [Xilinx XRT 2018.2 (Xilinx Run Time)](xrt.md)

1. Install Docker 
   
   - [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)  
   - [centos](https://docs.docker.com/install/linux/docker-ce/centos/#install-docker-ce)
     
   Note: Ensure /var/lib/docker has sufficient space (Should be > 5GB), if not move your Docker Root elsewhere.  
   
2. Download the appropriate ML Suite Container from xilinx.com
  
   - [ML Suite Caffe Container](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4.tar.gz)
   
3. Load the container
   ```
   # May need sudo
   $ docker load -i < xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4.tar.gz
   ```
   
4. Use the provided [script](../docker_run.sh) to launch and interact with the container
   ```
   # May need sudo
   $ ./docker_run.sh
  
   # The script by default will launch a container with the --rm flag, meaning after exiting, the container will be removed. 
   # Changes to source files will be lost.
   # A directory $MLSUITE_ROOT/share will be mounted in the container and can be used for easy file transfer between container and host.
   ```
  
5. Follow the Jupyter notebook or command line examples in the container

   [Jupyter Notebook Tutorials](../notebooks/README.md)
   - [Caffe Image Classification](../notebooks/image_classification_caffe.ipynb)
   - [Caffe Object Detection w/ YOLOv2](../notebooks/object_detection_yolov2.ipynb)
   Command Line Examples
   - [Caffe ImageNet Benchmark Models](../examples/caffe/README.md)
   - [Caffe VOC SSD Example](../examples/caffe/ssd-detect/README.md)
   - [Deployment Mode Examples](../examples/deployment_modes/README.md) 

6. Follow [Container Pipeline Example](container_pipeline.md) for example on how to stitch a container pipeline for preparing the model and running an inference server   
