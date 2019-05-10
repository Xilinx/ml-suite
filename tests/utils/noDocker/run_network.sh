#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash
MODEL_DIR=$1
DEPLOY=$2
CAFFEMODEL=$3
MEAN1=$4
MEAN2=$5
MEAN3=$6
SCALE=$7
compile_mode=$8
NW_NAME=$9
echo "compliler mode : ${compile_mode}"
echo "NW_NAME : $NW_NAME"
echo "MODEL_DIR = ${MODEL_DIR}"
echo "PLATFORM = ${PLATFORM}"
echo "Mean values : ${MEAN1} ${MEAN2} ${MEAN3}"
echo "scale : ${SCALE}"
echo "compliler mode : ${compile_mode}"
#exit 1;

mkdir output_logs/$NW_NAME
mkdir output_logs/$NW_NAME/$compile_mode
NW_LOG_DIR=output_logs/$NW_NAME/$compile_mode

if [ $# != 9 ]; then echo "failed to parse Usage : <MODEL DIR> <Deploy> <caffemodel> <Quant info> <mean values> <scale> <MODEL Name> <Compiler optim>"; exit 1; fi

# Env Setup
export MLSUITE_ROOT=../
export XFDNN_ROOT=$MLSUITE_ROOT
export XDNN_VERBOSE=1
export XBLAS_EMIT_PROFILING_INFO=$XDNN_VERBOSE
. ${MLSUITE_ROOT}/overlaybins/setup.sh ${PLATFORM}

# Network Details
PROTOTXT=${MODEL_DIR}/${DEPLOY}
CAFFEMODEL=${MODEL_DIR}/${CAFFEMODEL}
QUANT_CFGFILE=${MODEL_DIR}/quantize_info.txt
OUTPUT_DEEPHI_FILENAME=${MODEL_DIR}/new_quantize_info.txt

NUM_CLASSES=1000
DDR=256

# Compiler Details
XDNN_VERSION=3
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

CMD_FILE=${NW_LOG_DIR}/compiler_out_v${XDNN_VERSION}_deephi.cmd
WEIGHTS_DIR=${NW_LOG_DIR}/`basename ${CAFFEMODEL}`_data.h5


# Deployment Details
LABEL_FILE=${MLSUITE_ROOT}/examples/deployment_modes/synset_words.txt
IMAGE_DIR=/proj/mldata/DATA/ImageNet_ValSet_Orig
IMAGES=${MLSUITE_ROOT}/examples/deployment_modes/dog.jpg
#IMAGES=${MLSUITE_ROOT}/examples/classification/dog.jpg
GOLDEN_REF=${MLSUITE_ROOT}/examples/deployment_modes/gold.txt
IMAGE_DIR=${MLSUITE_ROOT}/models/data/ilsvrc12/ilsvrc12_img_val

run_compiler=1
if [ $run_compiler -eq 1 ];
then 
	#Remove the .h5 dir to avoid compiler errors 
	rm -rf ${CAFFEMODEL}_data.h5
    rm -f ${CMD_FILE}_quant.json
    rm -f ${CMD_FILE}.json

	echo "###### calling compiler ######"
    if [ "$compile_mode" = "nooptim" ];
    then
    	$python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    		-n ${PROTOTXT} \
    		-w ${CAFFEMODEL} \
    		-s all \
    		-m ${URAM_SIZE} \
    		-i ${DSP} \
    		-o ${MODEL_DIR}/res.png\
            --fromtensorflow \
            --anew ${MODEL_DIR}/optimized \
    		-g ${CMD_FILE}\
            --usedeephi \
            --quant_cfgfile ${QUANT_CFGFILE}\
    		-b 1 \
    		-C \
    	    -fwfc \
            -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
            
     elif [ "$compile_mode" = "standard" ]
     then
         if [ "$NW_NAME" = "resnet50_v2" ] || [ "$NW_NAME" = "resnet50_v1" ] || [ "$NW_NAME" = "resnet_50_v1_prune_round10_2.6G" ] \
             || [ "$NW_NAME" = "resnet_50_v1_prune_round5_3.7G" ] ;
         then
     	    $python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    	    	-n ${PROTOTXT} \
    	    	-w ${CAFFEMODEL} \
    	    	-s all \
    	    	-m ${URAM_SIZE} \
    	    	-i ${DSP} \
    	    	-o ${MODEL_DIR}/res.png\
                --fromtensorflow \
                --anew ${MODEL_DIR}/optimized \
    	    	-g ${CMD_FILE}\
                --usedeephi \
                --quant_cfgfile ${QUANT_CFGFILE}\
    	    	-b 1 \
    	    	-C \
    	        -fwfc \
    	        --pipelineconvmaxpool \
                --parallelread "['bottom','tops']" \
                --parallelismstrategy "['tops','bottom']" \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt
				compile_opt="pipelineconvmaxpool + parallelread + parallelismstrategy"
         else
     	    $python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    	    	-n ${PROTOTXT} \
    	    	-w ${CAFFEMODEL} \
    	    	-s all \
    	    	-m ${URAM_SIZE} \
    	    	-i ${DSP} \
    	    	-o ${MODEL_DIR}/res.png\
                --fromtensorflow \
                --anew ${MODEL_DIR}/optimized \
    	    	-g ${CMD_FILE}\
                --usedeephi \
                --quant_cfgfile ${QUANT_CFGFILE}\
    	    	-b 1 \
    	    	-C \
    	        -fwfc \
    	        --pipelineconvmaxpool \
                --parallelismstrategy "['tops','bottom']" \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				compile_opt="pipelineconvmaxpool + parallelismstrategy"
                #--parallelism \
                #--poolingaround \
         fi
     elif [ "$compile_mode" = "mix" ]
     then
         if [ "$NW_NAME" = "resnet50_v2" ] || [ "$NW_NAME" = "resnet50_v1" ] || [ "$NW_NAME" = "resnet_50_v1_prune_round10_2.6G" ] \
             || [ "$NW_NAME" = "resnet_50_v1_prune_round5_3.7G" ] ;
         then
     	    $python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    	    	-n ${PROTOTXT} \
    	    	-w ${CAFFEMODEL} \
    	    	-s all \
    	    	-m ${URAM_SIZE} \
    	    	-i ${DSP} \
    	    	-o ${MODEL_DIR}/res.png\
                --fromtensorflow \
                --anew ${MODEL_DIR}/optimized \
    	    	-g ${CMD_FILE}\
                --usedeephi \
                --quant_cfgfile ${QUANT_CFGFILE}\
    	    	-b 1 \
    	    	-C \
    	        -fwfc \
    	        --pipelineconvmaxpool \
                --parallelread "['bottom','tops']" \
                --parallelismstrategy "['tops','bottom']" \
				-mix \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt
				compile_opt="pipelineconvmaxpool + parallelread + parallelismstrategy + mix"
         else
     	    $python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    	    	-n ${PROTOTXT} \
    	    	-w ${CAFFEMODEL} \
    	    	-s all \
    	    	-m ${URAM_SIZE} \
    	    	-i ${DSP} \
    	    	-o ${MODEL_DIR}/res.png\
                --fromtensorflow \
                --anew ${MODEL_DIR}/optimized \
    	    	-g ${CMD_FILE}\
                --usedeephi \
                --quant_cfgfile ${QUANT_CFGFILE}\
    	    	-b 1 \
    	    	-C \
    	        -fwfc \
    	        --pipelineconvmaxpool \
                --parallelismstrategy "['tops','bottom']" \
				-mix \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				compile_opt="pipelineconvmaxpool + parallelismstrategy + mix"
                #--parallelism \
                #--poolingaround \
         fi            
     elif [ "$compile_mode" = "mix1" ]
     then
		 if [ "$NW_NAME" = "resnet50_v2" ] || [ "$NW_NAME" = "resnet50_v1" ] || [ "$NW_NAME" = "resnet_50_v1_prune_round10_2.6G" ] \
                           || [ "$NW_NAME" = "resnet_50_v1_prune_round5_3.7G" ]  ;
         then
     	    $python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
    	    	-n ${PROTOTXT} \
    	    	-w ${CAFFEMODEL} \
    	    	-s all \
    	    	-m ${URAM_SIZE} \
    	    	-i ${DSP} \
                --fromtensorflow \
                --anew ${NW_LOG_DIR}/optimized \
    	    	-g ${CMD_FILE}\
                --usedeephi \
                --quant_cfgfile ${QUANT_CFGFILE}\
    	    	-b 1 \
    	    	-C \
    	        -fwfc \
    	        --pipelineconvmaxpool \
                --parallelread "['bottom','tops']" \
                --parallelismstrategy "['tops','bottom']" \
                -mix \
                -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				compile_opt="pipelineconvmaxpool + parallelread + parallelismstrategy + mix"
         else
			# Mix mode enable
			$python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
				-n ${PROTOTXT} \
				-w ${CAFFEMODEL} \
				-s all \
				-m ${URAM_SIZE} \
				-i ${DSP} \
				--fromtensorflow \
				--anew ${NW_LOG_DIR}/optimized \
				-g ${CMD_FILE}\
				--usedeephi \
				--quant_cfgfile ${QUANT_CFGFILE}\
				-b 1 \
				-C \
				-fwfc \
				--pipelineconvmaxpool \
				--parallelism \
				--parallelismstrategy "['tops','bottom']" \
				-mix \
				-qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				compile_opt="pipelineconvmaxpool + parallelism + parallelismstrategy + mix"
				
                #--fancyreplications \
				#--customreplication "First Layer" \
				#-qz ${MODEL_DIR}/quantizer 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
				#--anew ${MODEL_DIR}/optimized \
				#-g ${MODEL_DIR}/compiler \
				#--fromtensorflow \
				#--pipelineconvmaxpool \
				#--customreplication "First Layer"\
				#--parallelismstrategy "['tops','bottom']" \
				#-mix \
				#--poolingaround \
				#--parallelism \
				#--dedicateddsp small \
				#rm $NW_LOG_DIR/compile_out.txt
		 fi
    else
            # Default  mode enable
			$python ${MLSUITE_ROOT}/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.py \
				-n ${PROTOTXT} \
				-w ${CAFFEMODEL} \
				-s all \
				-m ${URAM_SIZE} \
				-i ${DSP} \
				--fromtensorflow \
				--anew ${NW_LOG_DIR}/optimized \
				-g ${CMD_FILE}\
				--usedeephi \
				--quant_cfgfile ${QUANT_CFGFILE}\
				-b 1 \
				-C \
				-fwfc \
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

#exit 1

################# Generate throughput mode JSON
python throughput.py --i ${CMD_FILE}.json


################ Run Hardware 

# Define the arrays
array1=("${CMD_FILE}.json" "throughput.json")
array2=("latency" "throughput")

# get the length of the arrays
length=${#array1[@]}

printf "\n\n" >> nw_status.txt

echo "*** Network : $NW_NAME" >> nw_status.txt
printf "\n" >> nw_status.txt
echo "Run Directory Path        : $(pwd)" >> nw_status.txt
echo "Model Directory Path      : ${MODEL_DIR}" >> nw_status.txt
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

# do the loop
for ((i=0;i<$length;i++)); 
do
    echo "Run mode : ${array2[$i]}" >> nw_status.txt
    echo "${array1[$i]}"

    echo "###### Run Network on HW : ${MODEL_DIR} ######"
    $python ${MODEL_DIR}/test_classify.py \
           --images ${IMAGES} \
           --batch_sz 1 \
           --dsp ${DSP}  \
           --netcfg ${array1[$i]} \
           --quantizecfg ${CMD_FILE}_quant.json\
           --weights ${WEIGHTS_DIR} \
           --outsz ${NUM_CLASSES} \
           --labels ${LABEL_FILE} \
           --xclbin ${XCLBIN}\
       	   --img_mean $4 $5 $6 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_single_img_out.txt
    
    run_batch=1
    
    if [ $run_batch -eq 1 ];
    then
    $python ${MODEL_DIR}/test_classify.py \
    	   --images ${IMAGES} \
       	   --batch_sz 1 \
    	   --dsp ${DSP}  \
    	   --netcfg ${array1[$i]} \
    	   --quantizecfg ${CMD_FILE}_quant.json\
    	   --weights ${WEIGHTS_DIR} \
    	   --outsz ${NUM_CLASSES} \
    	   --labels ${LABEL_FILE} \
    	   --xclbin ${XCLBIN}\
           --img_mean $4 $5 $6 \
           --golden ${GOLDEN_REF}\
           --labels ${LABEL_FILE} \
           --images ${IMAGE_DIR} 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_batch_out.txt
    fi
    
    cat $NW_LOG_DIR/${array2[$i]}_single_img_out.txt | grep "hw_counter" >> nw_status.txt
    cat $NW_LOG_DIR/${array2[$i]}_single_img_out.txt | grep "exec_xdnn" >> nw_status.txt
    cat $NW_LOG_DIR/${array2[$i]}_batch_out.txt | grep "accuracy" >> nw_status.txt

done


echo "*** Network End" >> nw_status.txt 
printf "\n" >> nw_status.txt
