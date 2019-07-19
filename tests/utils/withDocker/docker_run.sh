#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
#!/usr/bin/env bash

tag=$1
mlsuite=`dirname $PWD`

#sudo \ 
docker run \
  --name "anup_container" \
  --rm \
  --net=host \
  --privileged=true \
  -it \
  -v /dev:/dev \
  -v /opt/xilinx:/opt/xilinx \
  -v $mlsuite:/opt/ml-suite \
  -v /wrk/acceleration/models:/opt/models \
  -v /wrk/acceleration/shareData:/opt/data \
  -w /opt/ml-suite \
  xilinxatg/ml_suite:$tag \
  bash
#&& cd test_deephi && bash -x nw_list.sh"

