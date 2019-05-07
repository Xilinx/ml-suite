## Getting Started with the ML Suite Docker Container
The ML Suite Docker container provides users with a largely self contained software environment for running inference on Xilinx FPGAs.
The only external dependencies are:  
- docker-ce
- Xilinx XRT (Xilinx Run Time)

1. Install Docker 
   
   - [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)  
     
   Note: Ensure /var/lib/docker has sufficient space (Should be > 5GB), if not move your Docker Root elsewhere.  
   
2. Launch an MLSuite Container using the Xilinx ML Suite Docker Image
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
   
   # Note: --rm will remove the container when you exit, omit this flag if you want the container to persist
   # Note: /dev and /opt/xilinx are mounted from the host to enable hardware runs; 
   #  Xilinx XRT must be installed on the host.
   ```

3. Follow the Jupyter notebook or command line examples in the container

   [Jupyter Notebook Tutorials](../notebooks/README.md)
   - [Caffe Image Classification](../notebooks/image_classification_caffe.ipynb)
   - [Caffe Object Detection w/ YOLOv2](../notebooks/object_detection_yolov2.ipynb)
   Command Line Examples
   - [Caffe ImageNet Benchmark Models](../examples/caffe/README.md)
   - [Caffe VOC SSD Example](../examples/caffe/ssd-detect/README.md)
   - [Deployment Mode Examples](../examples/deployment_modes/README.md) 

4. Follow [Container Pipeline Example](container_pipeline.md) for example on how to stitch a container pipeline for preparing the model and running an inference server   
