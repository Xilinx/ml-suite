#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "./run.sh --platform <platform> --config <config>"
  echo "./run.sh -p <platform> -c <config>"
  echo "Example:"
  echo "./run.sh -p alveo-u200 -c cfg/googlenet_v1.json"
}

CFG="cfg/googlenet_v1.json"
PLATFORM="alveo-u200"
VERBOSE=0

# Parse Options
OPTS=`getopt -o p:c:vh --long platform:,config:,verbose,help -n "$0" -- "$@"`
if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi
while true
do
  case "$1" in
    -p|--platform    ) PLATFORM="$2"         ; shift 2 ;;
    -c|--config      ) CFG="$2"              ; shift 2 ;;
    -v|--verbose     ) VERBOSE=1             ; shift 1 ;;
    -h|--help        ) usage                 ; exit  1 ;;
     *) break ;;
  esac
done


# Verbose Debug Profiling Prints
# Note, the VERBOSE mechanic here is not working
# Its always safer to set this manually
#export XBLAS_EMIT_PROFILING_INFO=1
# To be fixed
export XBLAS_EMIT_PROFILING_INFO=$VERBOSE

if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

. ${MLSUITE_ROOT}/overlaybins/setup.sh $PLATFORM

python benchmark.py --platform $PLATFORM --jsoncfg $CFG

