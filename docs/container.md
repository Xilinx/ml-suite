## Getting Started with the ML Suite Docker Container
The ML Suite Docker container provides users with a largely self contained software environment for running inference on Xilinx FPGAs.
The only external dependencies are:  
- docker-ce
- Xilinx XRT (Xilinx Run Time)

1. Install Docker 
   
   - [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)
    
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
