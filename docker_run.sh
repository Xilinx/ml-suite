#!/usr/bin/env bash
user=`whoami`

HERE=`dirname $(readlink -f $0)`

mkdir -p $HERE/share
chmod -R a+rwx $HERE/share

xclmgmt_driver="$(find /dev -name xclmgmt\*)"
docker_devices=""
echo "Found xclmgmt driver(s) at ${xclmgmt_driver}"
for i in ${xclmgmt_driver} ;
do
  docker_devices+="--device=$i "
done

render_driver="$(find /dev/dri -name renderD\*)"
echo "Found render driver(s) at ${render_driver}"
for i in ${render_driver} ;
do
  docker_devices+="--device=$i "
done

#sudo \ 
docker run \
  --rm \
  --net=host \
  --privileged=true \
  -it \
  $docker_devices \
  -v $HERE/share:/opt/ml-suite/share \
  -w /opt/ml-suite \
  xilinx-ml-suite-ubuntu-16.04-xrt-2018.2-caffe-mls-1.4:latest \
  bash
