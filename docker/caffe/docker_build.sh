#!/usr/bin/env bash
user=`whoami`

# Should give ubuntu:16.04 or ubuntu:18.04
image_name=${1:-ubuntu:16.04}

echo "Building docker for $image_name"

#sudo \
docker build \
  --network=host \
  -f Dockerfile \
  --build-arg IMAGE_NAME=$image_name \
  -t $user/ml-suite/$image_name \
  .
