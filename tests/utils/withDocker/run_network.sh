#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
#!/bin/bash

MODEL_PATH=$1
compile_mode=$2
NW_NAME="$(basename -- $MODEL_PATH)"

if [ $# != 2 ]; then echo "failed to parse Usage : <MODEL PATH> <compile mode>"; exit 1; fi

echo "NW_NAME : $NW_NAME"
echo "compliler mode : ${compile_mode}"
echo "PLATFORM = ${PLATFORM}"

# Log directory
mkdir output_logs/$NW_NAME
mkdir output_logs/$NW_NAME/$compile_mode
NW_LOG_DIR=output_logs/$NW_NAME/$compile_mode

echo "$NW_LOG_DIR"

# Env Setup
export XFDNN_ROOT=$MLSUITE_ROOT
export XDNN_VERBOSE=1
export XBLAS_EMIT_PROFILING_INFO=$XDNN_VERBOSE

# Must have this set in your shell, should point to root of ml-suite clone
if [ -z $MLSUITE_ROOT ]; then
      echo "Please set MLSUITE_ROOT, see you next time!"
      exit 1
fi

# Platform
if [ -z $PLATFORM ]; then
      echo "Please set PLATFORM, see you next time!"
      exit 1
fi

echo "### Setting up variables, auto-detect platform ###"
. ${MLSUITE_ROOT}/overlaybins/setup.sh ${PLATFORM}

# Compiler Details
XDNN_VERSION=3
DDR=256
BPP=1

# cmd JSON
CMD_FILE=${NW_LOG_DIR}/compiler_out_v${XDNN_VERSION}_deephi.cmd
WEIGHTS_DIR=${NW_LOG_DIR}/deploy.caffemodel_data.h5

if [[ ${XDNN_VERSION} == 3 ]]
then
    XCLBIN=${MLSUITE_ROOT}/overlaybins/${PLATFORM}/overlay_4.xclbin
    #XCLBIN=${MLSUITE_ROOT}/overlaybins/alveo-u200-ml/overlay_4.xclbin
    DSP=96
    URAM_SIZE=9
else
    DSP=56
    URAM_SIZE=6
    XCLBIN=${MLSUITE_ROOT}/overlaybins/${PLATFORM}/overlay_2.xclbin
fi


# Only necessary for development
# Externally only pyc is available
echo "### COMPILING PYTHON ###"
python -m compileall -f ${MLSUITE_ROOT}/xfdnn/tools

# Run Decent quantizer
export DECENT_DEBUG=1
echo "### Running DEEPHI Decent Quantizer ###"

run_decentq=1
if [ $run_decentq -eq 1 ];
then 
	if [ $NW_NAME != "inception_v2_ssd" ];
	then
	    $CAFFE_ROOT/build/tools/decent_q quantize -model ${MODEL_PATH}/${NW_NAME}_train_val.prototxt -weights ${MODEL_PATH}/${NW_NAME}.caffemodel -auto_test -test_iter 1 --calib_iter 1 2>&1 |tee $NW_LOG_DIR/decentq_out.txt
	else
	    $CAFFE_ROOT/build/tools/decent_q quantize -model ${MODEL_PATH}/${NW_NAME}_train_val.prototxt -weights ${MODEL_PATH}/${NW_NAME}.caffemodel -test_iter 1 --calib_iter 1 2>&1 |tee $NW_LOG_DIR/decentq_out.txt
	fi
fi

# copy the quantization files for debug purpose
mv quantize_results $NW_LOG_DIR/

echo "$NW_LOG_DIR/quantize_results/deploy.prototxt"
echo "$NW_LOG_DIR/quantize_results/deploy.caffemodel"

run_compiler=1
if [ $run_compiler -eq 1 ];
then 
    export GLOG_minloglevel=2 # Supress Caffe prints
	#Remove the .h5 dir to avoid compiler errors 
	rm -rf ${CAFFEMODEL}_data.h5
    rm -f ${CMD_FILE}_quant.json
    rm -f ${CMD_FILE}.json

	echo "###### calling compiler ######"
    if [ "$compile_mode" = "mix1" ]
    then
		 if [ "$NW_NAME" = "resnet50_v2" ] || [ "$NW_NAME" = "resnet50_v1" ] || [ "$NW_NAME" = "resnet_50_v1_prune_round10_2.6G" ] \
                           || [ "$NW_NAME" = "resnet_50_v1_prune_round5_3.7G" ]  ;
         then
     	    $python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    	    	-b ${BPP} \
    	    	-i ${DSP} \
    	    	-m ${URAM_SIZE} \
                -d ${DDR} \
    	    	-n $NW_LOG_DIR/quantize_results/deploy.prototxt \
    	    	-w $NW_LOG_DIR/quantize_results/deploy.caffemodel \
                --quant_cfgfile $NW_LOG_DIR/quantize_results/quantize_info.txt \
    	    	-g ${CMD_FILE}\
                --usedeephi \
    	    	-C \
    	        --pipelineconvmaxpool \
                --parallelread "['bottom','tops']" \
                --parallelismstrategy "['tops','bottom']" \
                -mix \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				compile_opt="pipelineconvmaxpool + parallelread + parallelismstrategy + mix"
         else
			# Mix mode enable
			$python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
                -b ${BPP} \
    	    	-i ${DSP} \
    	    	-m ${URAM_SIZE} \
                -d ${DDR} \
    	    	-n $NW_LOG_DIR/quantize_results/deploy.prototxt \
    	    	-w $NW_LOG_DIR/quantize_results/deploy.caffemodel \
                --quant_cfgfile $NW_LOG_DIR/quantize_results/quantize_info.txt \
    	    	-g ${CMD_FILE}\
                --usedeephi \
    	    	-C \
    	        --pipelineconvmaxpool \
				-mix \
				-qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				compile_opt="pipelineconvmaxpool + parallelism + parallelismstrategy + mix"
				#--parallelism \
                #--parallelismstrategy "['tops','bottom']" \
		 fi
    else
            # Default  mode enable
			$python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
                -b ${BPP} \
    	    	-i ${DSP} \
    	    	-m ${URAM_SIZE} \
                -d ${DDR} \
    	    	-n $NW_LOG_DIR/quantize_results/deploy.prototxt \
    	    	-w $NW_LOG_DIR/quantize_results/deploy.caffemodel \
                --quant_cfgfile $NW_LOG_DIR/quantize_results/quantize_info.txt \
    	    	-g ${CMD_FILE}\
                --usedeephi \
    	    	-C \
                --customreplication "First Layer" \
                --pipelineconvmaxpool \
                --parallelismstrategy "['tops','bottom']" \
                --poolingaround   \
                -mix  \
                --parallelism -PInc sn \
                --saveschedule $NW_LOG_DIR/nw.sched \
                --laodschedule $NW_LOG_DIR/nw.sched \
                --fancyreplication \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
                compile_opt="customreplication + pipelineconvmaxpool + parallelismstrategy + poolingaround + mix + parallelism + Pinc + fancyreplication"
    fi
fi

echo "######### Generate throughput mode JSON "
python $MLSUITE_ROOT/xfdnn/tools/compile/scripts/xfdnn_gen_throughput_json.py --i ${CMD_FILE}.json --o ${CMD_FILE}_tput.json 2>&1 |tee $NW_LOG_DIR/gen_tput_log.txt


################ Run Hardware 

# Define the arrays
array1=("${CMD_FILE}.json" "${CMD_FILE}_tput.json")
array2=("latency" "throughput")

# get the length of the arrays
length=${#array1[@]}

printf "\n\n" >> nw_status.txt

echo "*** Network : $NW_NAME" >> nw_status.txt
printf "\n" >> nw_status.txt
echo "Run Directory Path        : $(pwd)" >> nw_status.txt
echo "Model Directory Path      : ${MODEL_PATH}" >> nw_status.txt
echo "Output Log Directory Path : $(pwd)/$NW_LOG_DIR" >> nw_status.txt
echo "compile mode : $compile_opt" >> nw_status.txt

# check for any errors
ERR=$(cat $NW_LOG_DIR/compile_out.txt | grep "Error") 
STATUS=$(cat $NW_LOG_DIR/compile_out.txt | grep "SUCCESS True") 
echo "$ERR"

if [ -z "$STATUS" ];
then
	if [ ! -z "$ERR" ];
	then
	    echo "compiler error - $ERR" >> nw_status.txt
	    echo "check Output Log Directory Path for more details." >> nw_status.txt
	fi
fi

echo "Software int8 score : " >> nw_status.txt
grep "top-1" $NW_LOG_DIR/decentq_out.txt | tail -1 >> nw_status.txt
grep "top-5" $NW_LOG_DIR/decentq_out.txt | tail -1 >> nw_status.txt

# do the loop
for ((i=0;i<$length;i++)); 
do
    echo "Run mode : ${array2[$i]}" >> nw_status.txt
    echo "${array1[$i]}"

    echo "### Running MLSUITE Subgraph Cutter ###"
    python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_subgraph.py \
   	  --trainproto ${MODEL_PATH}/${NW_NAME}_train_val.prototxt \
    	  --inproto $NW_LOG_DIR/quantize_results/deploy.prototxt \
          --outproto $NW_LOG_DIR/xfdnn_${NW_NAME}_auto_cut.prototxt \
          --outtrainproto $NW_LOG_DIR/xfdnn_${NW_NAME}_auto_train_cut.prototxt \
          --cutAfter data \
          --xclbin $MLSUITE_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin \
          --netcfg ${array1[$i]} \
          --quantizecfg ${CMD_FILE}_quant.json \
          --weights ${WEIGHTS_DIR} \
          --profile True 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_xfdnn_subgrapph_out.txt


    #--trainproto ${MODEL_PATH}/${NW_NAME}_train_val.prototxt \
    #--trainproto $NW_LOG_DIR/quantize_results/deploy.prototxt \
          #--inproto ${MODEL_PATH}/${NW_NAME}_deploy.prototxt \
    #export XBLAS_EMIT_PROFILING_INFO=1
    echo "### Running $MODEL via pycaffe ###"
    python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_forward.py \
          --prototxt $NW_LOG_DIR/xfdnn_${NW_NAME}_auto_cut.prototxt \
          --caffemodel ${MODEL_PATH}/${NW_NAME}.caffemodel \
          --numBatches 100 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt

   
    echo "### Running $MODEL via pycaffe ###"
    python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_forward.py \
          --prototxt $NW_LOG_DIR/xfdnn_${NW_NAME}_auto_train_cut.prototxt \
          --caffemodel ${MODEL_PATH}/${NW_NAME}.caffemodel \
          --numBatches 100 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt

    grep "hw_counter" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
    grep "exec_xdnn" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
    grep "Average:" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -n 3 >> nw_status.txt
    #grep "top-1" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
    #grep "top-5" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
    #grep "hw_counter" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
done

echo "*** Network End" >> nw_status.txt
printf "\n" >> nw_status.txt
