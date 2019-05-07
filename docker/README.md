

## Building Xilinx's ml-suite container
### Install docker 
   
   - [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)  
     
   Note: Ensure /var/lib/docker has sufficient space (Should be > 5GB), if not move your Docker Root elsewhere. 

### Build and Run the caffe container
Currently only supporting ubuntu:16.04 and ubuntu:18.04 
```
cd /<PATH>/<TO>/ml-suite/docker/caffe

# Choose one of below
export BASE=ubuntu:16.04
export BASE=ubuntu:18.04

./docker_build.sh $BASE && ./docker_run.sh $BASE

# Note that once inside the container, you can use /opt/ml-suite/share to pass files back and forth between your host machine and the container.
```
### Build the tensorflow container
Coming soon... 
