

## Building Xilinx's ml-suite container
### Install docker 
   
   - [ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/#install-docker-ce)  
   - [centos](https://docs.docker.com/install/linux/docker-ce/centos/#install-docker-ce)
     
   Note: Ensure /var/lib/docker has sufficient space (Should be > 5GB), if not move your Docker Root elsewhere. 

### Build and Run the caffe container
```
cd /<PATH>/<TO>/ml-suite/docker/caffe

./docker_build.sh
./docker_run.sh

# Note that once inside the container, you can use /opt/ml-suite/share to pass files back and forth between your host machine and the container.
```
### Build the tensorflow container
Coming soon... 
