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

echo "Dump logs to dir : $NW_LOG_DIR"

printf "\n\n" >> nw_status.txt

echo "*** Network : $NW_NAME" >> nw_status.txt
printf "\n" >> nw_status.txt
echo "Run Directory Path        : $(pwd)" >> nw_status.txt
echo "Model Directory Path      : ${MODEL_PATH}" >> nw_status.txt
echo "Output Log Directory Path : $(pwd)/$NW_LOG_DIR" >> nw_status.txt


# Enable/Disable flags
run_decentq=1
run_compiler=1
run_tput_gen=1
run_sub_graph=1
run_hw=1
run_hw_batch=1
run_ssd_single_img=1
run_ssd_batch=1

# Env Setup
export XFDNN_ROOT=$MLSUITE_ROOT
export XDNN_VERBOSE=0
export XBLAS_EMIT_PROFILING_INFO=1

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

if [ $run_decentq -eq 1 ];
then 
	if [ $NW_NAME == "inception_v2_ssd" ];
	then
	    python $MLSUITE_ROOT/examples/caffe/ssd-detect/run_ssd.py --prototxt ${MODEL_PATH}/${NW_NAME}_train.prototxt --caffemodel ${MODEL_PATH}/${NW_NAME}.caffemodel --prepare 2>&1 |tee $NW_LOG_DIR/decentq_out.txt
	else
	    $CAFFE_ROOT/build/tools/decent_q quantize -model ${MODEL_PATH}/${NW_NAME}_train_val.prototxt -weights ${MODEL_PATH}/${NW_NAME}.caffemodel -auto_test -test_iter 1 --calib_iter 1 2>&1 |tee $NW_LOG_DIR/decentq_out.txt
	fi
fi

# copy the quantization files for debug purpose
mv quantize_results $NW_LOG_DIR/

echo "$NW_LOG_DIR/quantize_results/deploy.prototxt"
echo "$NW_LOG_DIR/quantize_results/deploy.caffemodel"

# check quantizer error status
DECENTQ_ERR=$(cat $NW_LOG_DIR/decentq_out.txt | grep "Check failed") 
echo "$DECENTQ_ERR"

if [ ! -z "$DECENTQ_ERR" ];
then
    echo "Quantizer/decentq error - $ERR" >> nw_status.txt
    echo "check Output Log Directory Path for more details." >> nw_status.txt
    echo "*** Network End" >> nw_status.txt
    exit 1;
fi


echo "Software int8 score : " >> nw_status.txt
grep "top-1" $NW_LOG_DIR/decentq_out.txt | tail -1 >> nw_status.txt
grep "top-5" $NW_LOG_DIR/decentq_out.txt | tail -1 >> nw_status.txt


if [ $run_compiler -eq 1 ];
then 
    export GLOG_minloglevel=2 # Supress Caffe prints

    #Remove the .h5 dir to avoid compiler errors 
	rm -rf $NW_LOG_DIR/deploy.caffemodel_data.h5
    rm -f ${CMD_FILE}_quant.json
    rm -f ${CMD_FILE}.json

	echo "###### calling compiler ######"
    if [ "$compile_mode" = "default" ];
    then
	    if [ "$NW_NAME" = "inception_v4" ] || [ "$NW_NAME" = "inception_v3" ];
	    then
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
                    --fancyreplication \
                    -mix  \
                    --parallelism -PInc sn \
                    --saveschedule $NW_LOG_DIR/nw.sched \
                    --laodschedule $NW_LOG_DIR/nw.sched \
                    -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
                    compile_opt="customreplication + pipelineconvmaxpool + parallelismstrategy + poolingaround + \
                        mix + parallelism + Pinc + fancyreplication"

	    elif [ "$NW_NAME" = "inception_v2_ssd" ];
	    then
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
        	    	-qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
	            	compile_opt="pipelineconvmaxpool"

	    elif [ "$NW_NAME" = "resnet50_v2" ];
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
	    	        --inputcut "data_bn" \
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
                    -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
                    compile_opt="customreplication + pipelineconvmaxpool + parallelismstrategy + poolingaround + \
                        mix + parallelism + Pinc"
            
	    else			 
            # For all other networks
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
                    --parallelism -PInc sn \
                    --saveschedule $NW_LOG_DIR/nw.sched \
                    --laodschedule $NW_LOG_DIR/nw.sched \
                    --customreplication "First Layer" \
        	    	--parallelismstrategy "['tops','bottom']" \
                    --poolingaround   \
                    -mix  \
                    -qz "" 2>&1 |tee $NW_LOG_DIR/compile_out.txt 
                    compile_opt="customreplication + pipelineconvmaxpool + parallelismstrategy + poolingaround + \
                        mix + parallelism + Pinc"
            fi
    fi
fi


# update compiler details to log
echo "compile mode : $compile_opt" >> nw_status.txt

# check compiler error status
ERR=$(cat $NW_LOG_DIR/compile_out.txt | grep "Error" | tail -1) 
STATUS=$(cat $NW_LOG_DIR/compile_out.txt | grep "SUCCESS True") 
echo "$ERR"

if [ -z "$STATUS" ];
then
	if [ ! -z "$ERR" ];
	then
	    echo "compiler error - $ERR" >> nw_status.txt
	    echo "check Output Log Directory Path for more details." >> nw_status.txt
        echo "*** Network End" >> nw_status.txt
        exit 1;
	fi
fi



if [ $run_tput_gen -eq 1 ];
then

    echo "######### Generate throughput mode JSON "
    python $MLSUITE_ROOT/xfdnn/tools/compile/scripts/xfdnn_gen_throughput_json.py --i ${CMD_FILE}.json --o ${CMD_FILE}_tput.json 2>&1 |tee $NW_LOG_DIR/gen_tput_log.txt

fi

################ Run Hardware 

# Define the arrays
array1=("${CMD_FILE}.json" "${CMD_FILE}_tput.json")
array2=("latency" "throughput")

# get the length of the arrays
length=${#array1[@]}

if [ $run_hw -eq 1 ];
then

    # do the loop
    for ((i=0;i<$length;i++)); 
    do
        echo "Run mode : ${array2[$i]}" >> nw_status.txt
        echo "${array1[$i]}"
    
        if [ $run_sub_graph -eq 1 ];
        then
        	if [ $NW_NAME != "inception_v2_ssd" ];
        	then
    	    	proto_ext=_val
    	    fi
    
       	    echo "### Running MLSUITE Subgraph Cutter ###"
    	    python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_subgraph.py \
    	          --trainproto ${MODEL_PATH}/${NW_NAME}_train${proto_ext}.prototxt \
    	    	  --inproto $NW_LOG_DIR/quantize_results/deploy.prototxt \
    	          --outproto $NW_LOG_DIR/${array2[$i]}_xfdnn_${NW_NAME}_auto_cut.prototxt \
    	          --outtrainproto $NW_LOG_DIR/${array2[$i]}_xfdnn_${NW_NAME}_auto_train_cut.prototxt \
    	          --cutAfter data \
    	          --xclbin $MLSUITE_ROOT/overlaybins/$MLSUITE_PLATFORM/overlay_4.xclbin \
    	          --netcfg ${array1[$i]} \
    	          --quantizecfg ${CMD_FILE}_quant.json \
    	          --weights ${WEIGHTS_DIR} \
    	          --profile True 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_xfdnn_subgrapph_out.txt
    	    echo "### Running MLSUITE Subgraph Cutter Done ###"
            fi
    
            ######################### Run hardware
            if [ $NW_NAME = "inception_v2_ssd" ];
            then
    	        echo "### Running SSD like detect ###"
    
                if [ $run_ssd_single_img -eq 1 ];
    	        then
    	    	    echo "### Running SSD like detect for single image ###"
    	    	    # single image
    	    	    python $MLSUITE_ROOT/examples/caffe/ssd-detect/run_ssd.py \
    	    	    	--prototxt $NW_LOG_DIR/${array2[$i]}_xfdnn_${NW_NAME}_auto_cut.prototxt \
    	    	    	--caffemodel $NW_LOG_DIR/quantize_results/deploy.caffemodel \
    	    	    	--labelmap_file ${MLSUITE_ROOT}/examples/caffe/ssd-detect/labelmap_voc_19c.prototxt \
            	   	 	--image $MLSUITE_ROOT/examples/caffe/ssd-detect/test_pic/000022.jpg \
                            2>&1 |tee $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt
    	    	    
                    mv res_det.jpg $NW_LOG_DIR/
                    
                    SING_SSD_ERR=$(grep "ERROR" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1)
                    if [ ! -z "$SING_SSD_ERR" ];
    	            then
    	                echo "HW Error : $SING_SSD_ERR" >> nw_status.txt
    	                echo "check Output Log Directory Path for more details." >> nw_status.txt
                        echo "*** Network End" >> nw_status.txt
                        exit 1;
    	            fi
    

    	        fi
    
    	
                if [ $run_ssd_batch -eq 1 ];
    	        then
    		        # Multiple images
    		        python $MLSUITE_ROOT/examples/caffe/ssd-detect/run_ssd.py \
    		        	--prototxt $NW_LOG_DIR/${array2[$i]}_xfdnn_${NW_NAME}_auto_cut.prototxt \
    		        	--caffemodel $NW_LOG_DIR/quantize_results/deploy.caffemodel \
    		        	--labelmap_file ${MLSUITE_ROOT}/examples/caffe/ssd-detect/labelmap_voc_19c.prototxt \
        	       	 	--test_image_root ${SSD_DATA}/VOCdevkit/VOC2007/JPEGImages/ \
        	       		--image_list_file ${SSD_DATA}/VOCdevkit/VOC2007/ImageSets/Main/test.txt \
    		        	--compute_map_script_path $MLSUITE_ROOT/examples/caffe/ssd-detect/evaluation_py2.py \
    		           	--gt_file ${SSD_DATA}/voc07_gt_file_19c.txt --validate 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt

                    BATCH_SSD_ERR=$(grep "ERROR" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -1)
                    if [ ! -z "$BATCH_SSD_ERR" ];
    	            then
    	                echo "HW Error : $BATCH_SSD_ERR" >> nw_status.txt
    	                echo "check Output Log Directory Path for more details." >> nw_status.txt
                        echo "*** Network End" >> nw_status.txt
                        exit 1;
    	            fi
                fi
        	
    	        grep "hw_counter" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
        	    grep "exec_xdnn" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
            	grep "mAP:" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
    
    	        echo "### Running SSD like detect Done ###"
    
            else

    	        echo "### Running classification network ###"
        	    #export XBLAS_EMIT_PROFILING_INFO=1
        	    echo "### Running $MODEL via pycaffe ###"
        	    python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_forward.py \
        	          --prototxt $NW_LOG_DIR/${array2[$i]}_xfdnn_${NW_NAME}_auto_cut.prototxt \
        	          --caffemodel ${MODEL_PATH}/${NW_NAME}.caffemodel \
        	          --numBatches 1 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt
    
                SING_HW_ERR=$(grep "ERROR" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1)
                if [ ! -z "$SING_HW_ERR" ];
    	        then
    	            echo "HW Error : $SING_HW_ERR" >> nw_status.txt
    	            echo "check Output Log Directory Path for more details." >> nw_status.txt
                    echo "*** Network End" >> nw_status.txt
                    exit 1;
    	        fi
    
    	        if [ $run_hw_batch -eq 1 ];
                then   
    	    	    echo "### Running $MODEL via pycaffe ###"
        		    python $MLSUITE_ROOT/xfdnn/rt/scripts/framework/caffe/xfdnn_forward.py \
        		      --prototxt $NW_LOG_DIR/${array2[$i]}_xfdnn_${NW_NAME}_auto_train_cut.prototxt \
        		      --caffemodel ${MODEL_PATH}/${NW_NAME}.caffemodel \
            	      --numBatches 10 2>&1 |tee $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt
    	        fi
        	
                BATCH_HW_ERR=$(grep "ERROR" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -1)
                if [ ! -z "$BATCH_HW_ERR" ];
    	        then
    	            echo "HW Error : $BATCH_HW_ERR" >> nw_status.txt
    	            echo "check Output Log Directory Path for more details." >> nw_status.txt
                    echo "*** Network End" >> nw_status.txt
                    exit 1;
    	        fi
       
        
    	        grep "hw_counter" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
        	    grep "exec_xdnn" $NW_LOG_DIR/${array2[$i]}_single_img_xfdnn_fwd_out.txt | tail -1 >> nw_status.txt
        	    grep "Average:" $NW_LOG_DIR/${array2[$i]}_xfdnn_fwd_out.txt | tail -n 3 >> nw_status.txt
            fi
    done
fi

echo "*** Network End" >> nw_status.txt
printf "\n" >> nw_status.txt
