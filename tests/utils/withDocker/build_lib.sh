#!/bin/bash

# Export MLSuite path
if [ -z $MLSUITE_ROOT ];
then
    export MLSUITE_ROOT=/wrk/acceleration/users/anup/MLsuite_mastr
    echo "##### Setting default path as : $MLSUITE_ROOT. Please set to required path"
    #exit 1;
fi

# pull latest 
git pull -r

# pull lfs files
export PATH=$PATH:/wrk/acceleration/MLsuite_Embedded/anup/gitlfs/
git lfs pull

# Enable below to build xdnn lib on CentOS 7.4
#export PATH=/tools/batonroot/rodin/devkits/lnx64/binutils-2.26/bin:/tools/batonroot/rodin/devkits/lnx64/make-4.1/bin:/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/bin:$PATH
#export LD_LIBRARY_PATH=/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/lib64:$LD_LIBRARY_PATH

# Build rt
cd $MLSUITE_ROOT/xfdnn/rt/xdnn_cpp
make clean;make -j8
cd -

#Build nms for yolo
cd $MLSUITE_ROOT/apps/yolo/nms
make
cd -

