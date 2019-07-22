#!/usr/bin/env bash

#tag=$1

tag=ubuntu-16.04-xrt-2018.2-caffe-mls-1.4


# Export MLSuite path
if [ -z $MLSUITE_ROOT ];
then
    export MLSUITE_ROOT=/wrk/acceleration/users/anup/MLsuite_mastr
    echo "##### Setting default path as : $MLSUITE_ROOT. Please set to required path"
    #exit 1;
fi


#sudo \ 
docker run \
  --name "mluser_container" \
  --rm \
  --net=host \
  --privileged=true \
  -it \
  -d \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -v $MLSUITE_ROOT:/opt/ml-suite \
  -v /wrk/acceleration/shareData:/opt/data \
  -v $MLSUITE_ROOT/share:/opt/share \
  -v $MLSUITE_ROOT/share/CK-TOOLS:/home/mluser/CK-TOOLS \
  -w /opt/ml-suite \
  xilinxatg/ml_suite:$tag \
  bash

