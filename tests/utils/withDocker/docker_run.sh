#!/usr/bin/env bash

tag=$1

# export MLSUITE_ROOT
if [ -z $MLSUITE_ROOT];
then
    echo "Please set MLSUITE_ROOT"
    exit 1;
fi


#sudo \ 
docker run \
  --name "mluser_container" \
  --rm \
  --net=host \
  --privileged=true \
  -it \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -v $MLSUITE_ROOT:/opt/ml-suite \
  -v /wrk/acceleration/shareData:/opt/data \
  -v $MLSUITE_ROOT/share:/opt/share \
  -v $MLSUITE_ROOT/share/CK-TOOLS:/home/mluser/CK-TOOLS \
  -w /opt/ml-suite \
  xilinxatg/ml_suite:$tag \
  bash

