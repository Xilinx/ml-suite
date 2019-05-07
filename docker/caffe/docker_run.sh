#!/usr/bin/env bash
user=`whoami`

image_name=${1:-ubuntu:16.04}

HERE=`dirname $(readlink -f $0)`

mkdir -p $HERE/share
chmod -R a+rwx $HERE/share

#sudo \ 
docker run \
  --rm \
  --net=host \
  --privileged=true \
  -it \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -v $HERE/share:/opt/ml-suite/share \
  -w /opt/ml-suite \
  $user/ml-suite/$image_name \
  bash
