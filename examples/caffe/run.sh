#!/bin/bash

echo "### Cleaning Stale Files From Previous Run ###"
./clean.sh
mkdir work

# Must have this set in your shell, should point to root of ml-suite clone
if [ -z $MLSUITE_ROOT ]; then
  echo "Please set MLSUITE_ROOT, see you next time!"
  exit 1
fi

# Only necessary for development
# Externally only pyc is available
echo "### COMPILING PYTHON ###"
python -m compileall -f ${MLSUITE_ROOT}/xfdnn/tools

# User must pass the <path>/basename of the model
# i.e. resnet50_without_bn is the basename of the prototxt "resnet50_without_bn_deploy.prototxt"
# example: ./run.sh /opt/models/caffe/resnet50/resnet50_without_bn 
if [ -z $1 ]; then
  echo "Please provide MODEL path, see you next time!"
  exit 1
fi
MODEL_PATH=$1
MODEL="$(basename -- $MODEL_PATH)"

echo "### Setting up variables, auto-detect platform ###"
. ${MLSUITE_ROOT}/overlaybins/setup.sh "" 

export DECENT_DEBUG=1
echo "### Running DEEPHI Decent Quantizer ###"
$CAFFE_ROOT/build/tools/decent_q quantize -model ${MODEL_PATH}_train_val.prototxt -weights ${MODEL_PATH}.caffemodel -auto_test -test_iter 1 --calib_iter 1

# Compiler Args
BPP=1
DSP_WIDTH=96
MEM=9
DDR=256
export GLOG_minloglevel=2 # Supress Caffe prints
echo "### Running MLSUITE Compiler ###"
python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.pyc \
  -b ${BPP} \
  -i ${DSP_WIDTH} \
  -m ${MEM} \
  -d ${DDR} \
  -mix \
  --pipelineconvmaxpool \
  --usedeephi \
  --quant_cfgfile quantize_results/quantize_info.txt \
  -n quantize_results/deploy.prototxt \
  -w quantize_results/deploy.caffemodel \
  -g work/compiler \
  -qz work/quantizer \
  -C

echo "### Running MLSUITE Subgraph Cutter ###"
python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_subgraph.py \
  --inproto quantize_results/deploy.prototxt \
  --trainproto ${MODEL_PATH}_train_val.prototxt \
  --outproto xfdnn_${MODEL}_auto_cut.prototxt \
  --cutAfter data \
  --xclbin $MLSUITE_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin \
  --netcfg work/compiler.json \
  --quantizecfg work/quantizer.json \
  --weights work/deploy.caffemodel_data.h5 \
  --profile True

#export XBLAS_EMIT_PROFILING_INFO=1
echo "### Running $MODEL via pycaffe ###"
python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_forward.py \
  --prototxt xfdnn_${MODEL}_auto_cut.prototxt \
  --caffemodel ${MODEL_PATH}.caffemodel \
  --numBatches 10
