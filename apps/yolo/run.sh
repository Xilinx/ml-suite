#!/usr/bin/env bash
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
#!/usr/bin/env bash

usage() {
  echo "Usage:"
  echo "./run.sh --platform <platform> --test <test> --model <model> --kcfg <kcfg> --bitwidth <bitwidth>"
  echo "./run.sh  -p        <platform>  -t    <test>  -m <model>  -k <kcfg>  -b <bitwidth>"
  echo "<platform> : 1525 / 1525-ml / alveo-u200 / alveo-u200-ml / alveo-u250 / aws / nimbix"
  echo "<test>     : test_classify / streaming_classify"
  echo "<kcfg>     : med / large / v3"
  echo "<bitwidth> : 8 / 16"
  echo "<compilerOpt> : autoAllOpt / latency / throughput"
  echo "Some tests require a directory of images to process."
  echo "To process a directory in a non-standard location use -d <directory> or --directory <directory>"
  echo "Some tests require a batchSize argument to know how many images to load simultaneously."
  echo "To provide batchSize use --batchsize <batchsize>"
  echo "-c allows to choose compiler optimization, for example, latency or throughput or autoAllOptimizations."
  echo "-g runs entire test providing top-1 and top-5 results"

}


# Default
TEST="test_classify"
MODEL="googlenet_v1"
KCFG="large"
BITWIDTH="8"
ACCELERATOR="0"
BATCHSIZE=-1
VERBOSE=0
ZELDA=0
PERPETUAL=0
IMG_INPUT_SCALE=1.0
# These variables are used in case there multiple FPGAs running in parallel
NUMDEVICES=1
DEVICEID=0
NUMPREPPROC=4
COMPILEROPT="autoAllOpt.json"
# Parse Options
OPTS=`getopt -o p:t:m:k:b:d:s:a:n:i:c:y:gvzxh --long platform:,test:,model:,kcfg:,bitwidth:,directory:,numdevices:,deviceid:,batchsize:,compilerOpt:,numprepproc,checkaccuracy,verbose,zelda,perpetual,help -n "$0" -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; usage; exit 1 ; fi
  
while true
do
  case "$1" in
    -p |--platform      ) MLSUITE_PLATFORM="$2" ; shift 2 ;;
    -t |--test          ) TEST="$2"             ; shift 2 ;;
    -m |--model         ) MODEL="$2"            ; shift 2 ;;
    -k |--kcfg          ) KCFG="$2"             ; shift 2 ;;
    -b |--bitwidth      ) BITWIDTH="$2"         ; shift 2 ;;
    -d |--directory     ) DIRECTORY="$2"        ; shift 2 ;;
    -s |--batchsize     ) BATCHSIZE="$2"        ; shift 2 ;;
    -a |--accelerator   ) ACCELERATOR="$2"      ; shift 2 ;;
    -n |--numdevices    ) NUMDEVICES="$2"       ; shift 2 ;;
    -i |--deviceid      ) DEVICEID="$2"         ; shift 2 ;;
    -c |--compilerOpt   ) COMPILEROPT="$2"      ; shift 2 ;;
    -y |--numprepproc   ) NUMPREPPROC="$2"      ; shift 2 ;;
    -g |--checkaccuracy ) GOLDEN="$2"           ; shift 2 ;;
    -v |--verbose       ) VERBOSE=1             ; shift 1 ;;
    -z |--zelda         ) ZELDA=1               ; shift 1 ;;
    -x |--perpetual     ) PERPETUAL=1           ; shift 1 ;;
    -cn|--customnet     ) CUSTOM_NETCFG="$2"    ; shift 2 ;;
    -cq|--customquant   ) CUSTOM_QUANTCFG="$2"  ; shift 2 ;;
    -cw|--customwts     ) CUSTOM_WEIGHTS="$2"   ; shift 2 ;;
    -h |--help          ) usage                 ; exit  1 ;;
     *) break ;;
  esac
done

echo -e $COMPILEROPT
# Verbose Debug Profiling Prints
# Note, the VERBOSE mechanic here is not working
# Its always safer to set this manually
export XBLAS_EMIT_PROFILING_INFO=1
# To be fixed
#export XBLAS_EMIT_PROFILING_INFO=$VERBOSE
export XDNN_VERBOSE=$VERBOSE
# Set Platform Environment Variables
if [ -z $MLSUITE_ROOT ]; then
  MLSUITE_ROOT=../..
fi

# Build Non-Max Suppression C-code
cd nms
make 
cd ..

if [ -z $CAFFE_ROOT ]; then
    CAFFE_ROOT=/wrk/acceleration/users/arun/caffe
fi

if [ "$MLSUITE_PLATFORM" == "gpu" ]; then
  echo -e "GPU mode selected"
  echo -e "check for installation of cuda and cudnn"
  echo -e "Printing CUDA_HOME"
  echo $CUDA_HOME
  echo -e "Printing PATH"
  echo $PATH
  echo -e "Printing LD_LIBRARY_PATH"
  echo $LD_LIBRARY_PATH
  
  if [ -z $CAFFE_BACKEND_ROOT ]; then
    CAFFE_BACKEND_ROOT=../../../MLretraining/
  fi
else
  . ${MLSUITE_ROOT}/overlaybins/setup.sh ${MLSUITE_PLATFORM}
fi

# Determine FPGAOUTSZ which depend upon model
if [ "$MODEL" == "yolo_v2_224" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_deploy_224.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "yolo_v2_416" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_deploy_416.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=416
  INSHAPE_HEIGHT=416
elif [ "$MODEL" == "yolo_v2_608" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_deploy_608.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=608
  INSHAPE_HEIGHT=608
elif [ "$MODEL" == "yolo_v2_tiny_224" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_224.prototxt
  NET_DEF_FPGA=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_224_fpga.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "yolo_v2_tiny_416" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_416.prototxt
  NET_DEF_FPGA=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_416_fpga.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=416
  INSHAPE_HEIGHT=416
elif [ "$MODEL" == "yolo_v2_tiny_608" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_608.prototxt
  NET_DEF_FPGA=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_tiny_608_fpga.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=608
  INSHAPE_HEIGHT=608
elif [ "$MODEL" == "yolo_v2_standard_224" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_224.prototxt
  NET_DEF_FPGA=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_224_fpga.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "yolo_v2_standard_416" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_416.prototxt
  NET_DEF_FPGA=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_416_fpga.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=416
  INSHAPE_HEIGHT=416
elif [ "$MODEL" == "yolo_v2_standard_608" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_608.prototxt
  NET_DEF_FPGA=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_standard_608_fpga.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=608
  INSHAPE_HEIGHT=608
elif [ "$MODEL" == "yolo_v2_prelu_224" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_prelu_224.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=224
  INSHAPE_HEIGHT=224
elif [ "$MODEL" == "yolo_v2_prelu_416" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_prelu_416.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=416
  INSHAPE_HEIGHT=416
elif [ "$MODEL" == "yolo_v2_prelu_608" ]; then
  NET_DEF=${MLSUITE_ROOT}/models/caffe/yolov2/fp32/yolo_v2_prelu_608.prototxt
  FPGAOUTSZ=2048000
  INSHAPE_CHANNELS=3
  INSHAPE_WIDTH=608
  INSHAPE_HEIGHT=608
fi

#FPGAOUTSZ=2048000 # default to something large enough to store any layer

# Determine XCLBIN and DSP_WIDTH
XCLBIN="not_found.xclbin"
WEIGHTS=./data/${MODEL}_data
if [ "$KCFG" == "med" ]; then
  DSP_WIDTH=28
  XCLBIN=overlay_1.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_0.xclbin
  fi
  NETCFG=./data/${MODEL}_${DSP_WIDTH}.json
  QUANTCFG=./data/${MODEL}_${BITWIDTH}b.json
elif [ "$KCFG" == "large" ]; then
  DSP_WIDTH=56
  XCLBIN=overlay_3.xclbin
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_2.xclbin
  fi
  NETCFG=./data/${MODEL}_${DSP_WIDTH}.json
  QUANTCFG=./data/${MODEL}_${BITWIDTH}b.json
elif [ "$KCFG" == "v3" ]; then
  DSP_WIDTH=96
  MEMORY=9
  if [ "$BITWIDTH" == "8" ]; then
    XCLBIN=overlay_4.xclbin
  elif [ "$BITWIDTH" == "16" ]; then
    XCLBIN=overlay_5.xclbin
  fi


elif [ "$MLSUITE_PLATFORM" == "gpu" ] ; then
  echo -e "Running in GPU mode, no XDNN config required "  
else
  echo "Unsupported kernel config $KCFG"
  exit 1
fi

YOLO_TYPE="none"
LABELS=./synset_words.txt
if [ $MODEL == "yolo_v2_224" ] || [ $MODEL == "yolo_v2_416" ] || [ $MODEL == "yolo_v2_608" ] || 
   [ $MODEL == "yolo_v2_tiny_224" ] || [ $MODEL == "yolo_v2_tiny_416" ] || [ $MODEL == "yolo_v2_tiny_608" ] ||
   [ $MODEL == "yolo_v2_standard_224" ] || [ $MODEL == "yolo_v2_standard_416" ] || [ $MODEL == "yolo_v2_standard_608" ] ||
   [ $MODEL == "yolo_v2_prelu_224" ] || [ $MODEL == "yolo_v2_prelu_416" ] || [ $MODEL == "yolo_v2_prelu_608" ]; then
# setting dummy paths for below  
  NETCFG=$NET_DEF
#'work/yolo.cmds'
  WEIGHTS=$NET_DEF
#'work/yolov2.caffemodel_data'
  QUANTCFG=$NET_DEF  
  
  NUM_CLASSES=80
  LABELS='./coco.names'
  
  if [ $MODEL == "yolo_v2_224" ] || [ $MODEL == "yolo_v2_416" ] || [ $MODEL == "yolo_v2_608" ] ; then
    NET_WEIGHTS=../../models/caffe/yolov2/fp32/yolov2.caffemodel
    NET_DEF_FPGA=$NET_DEF
    YOLO_TYPE="xilinx_yolo_v2"
  elif  [ $MODEL == "yolo_v2_tiny_224" ] || [ $MODEL == "yolo_v2_tiny_416" ] || [ $MODEL == "yolo_v2_tiny_608" ] ; then
    NET_WEIGHTS=../../models/caffe/yolov2/fp32/yolo_v2_tiny.caffemodel
    YOLO_TYPE="tiny_yolo_v2"
  elif  [ $MODEL == "yolo_v2_standard_224" ] || [ $MODEL == "yolo_v2_standard_416" ] || [ $MODEL == "yolo_v2_standard_608" ]; then
    NET_WEIGHTS=../../models/caffe/yolov2/fp32/yolo_v2_standard.caffemodel
    YOLO_TYPE="standard_yolo_v2"
  elif [ $MODEL == "yolo_v2_prelu_224" ] || [ $MODEL == "yolo_v2_prelu_416" ] || [ $MODEL == "yolo_v2_prelu_608" ]; then
    NET_WEIGHTS=../../models/caffe/yolov2/fp32/yolo_v2_prelu.caffemodel
    YOLO_TYPE="xilinx_prelu_yolo_v2"
    NET_DEF_FPGA=$NET_DEF
  fi
  
  if [ "$KCFG" == "v3" ]; then
    export DECENT_DEBUG=1
    #: > images.txt
    #ls -d -1 "$MLSUITE_ROOT/"xfdnn/tools/quantize/calibration_directory/*.jpg >> images.txt
    #sed -e 's/$/ 1/' -i images.txt 
    DUMMY_PTXT=dummy.prototxt
    IMGLIST="$MLSUITE_ROOT/"apps/yolo/images.txt
    CALIB_DATASET="$MLSUITE_ROOT/"apps/yolo/test_image_set
    python get_decent_q_prototxt.py ${CAFFE_ROOT}/python/ $NET_DEF_FPGA  $DUMMY_PTXT $IMGLIST  $CALIB_DATASET
    ${CAFFE_ROOT}/build/tools/decent_q quantize  -model $DUMMY_PTXT -weights $NET_WEIGHTS --output_dir work/  -calib_iter 100
    
    # Compiler Args
    BPP=1
    DSP_WIDTH=96
    MEM=9
    DDR=256
    export GLOG_minloglevel=2 # Supress Caffe prints
    echo "### Running MLSUITE Compiler ###"

    COMPILER_BASE_OPT=" -b ${BPP} \
      -i ${DSP_WIDTH} \
      -m ${MEM} \
      -d ${DDR} \
      --usedeephi \
      --quant_cfgfile work/quantize_info.txt \
      -n work/deploy.prototxt \
      -w work/deploy.caffemodel \
      -g work/compiler \
      -qz work/quantizer \
      -C "

    if [ $COMPILEROPT == "latency" ] || [ $COMPILEROPT == "throughput" ]; then
       COMPILER_BASE_OPT+="-mix --poolingaround -P "  
       COMPILER_BASE_OPT+="-pcmp --parallelread ['bottom','tops'] -Q ['tops','bottom'] "
    fi

    python $MLSUITE_ROOT/xfdnn/tools/compile/bin/xfdnn_compiler_caffe.pyc $COMPILER_BASE_OPT
    echo -e $COMPILEROPT  
    NETCFG=work/compiler.json
    QUANTCFG=work/quantizer.json
    WEIGHTS=work/deploy.caffemodel_data

    if [ $COMPILEROPT == "throughput" ]; then
       python $MLSUITE_ROOT/xfdnn/tools/compile/scripts/xfdnn_gen_throughput_json.py --i work/compiler.json --o work/compiler_tput.json            
       NETCFG=work/compiler_tput.json
    fi  
    
  fi  
     
fi  
  
if [ ! -z $CUSTOM_NETCFG ]; then
  echo -e "the netconfig file name to have a x.cmd.json format"
  echo -e "if customwts are not given compiler will be run to generate this "
  NETCFG=$CUSTOM_NETCFG
fi
if [ ! -z $CUSTOM_WEIGHTS ]; then
  if [ -z $CUSTOM_NETCFG ]; then
    echo -e "cannot give weights without customnet/-cn file"
    exit 1
  fi
  WEIGHTS=$CUSTOM_WEIGHTS
fi
if [ ! -z $CUSTOM_QUANTCFG ]; then
  echo -e "customquant file name to have a x.json format"
  QUANTCFG=$CUSTOM_QUANTCFG
fi

echo -e "Running:\n Test: $TEST\n Model: $MODEL\n Platform: $MLSUITE_PLATFORM\n Xclbin: $XCLBIN\n Kernel Config: $KCFG\n Precision: $BITWIDTH\n Accelerator: $ACCELERATOR\n"

BASEOPT="--xclbin $XCLBIN_PATH/$XCLBIN 
         --netcfg $NETCFG 
         --weights $WEIGHTS 
         --labels $LABELS 
         --quantizecfg $QUANTCFG 
         --img_input_scale $IMG_INPUT_SCALE 
         --batch_sz $BATCHSIZE 
         --in_shape $INSHAPE_CHANNELS $INSHAPE_WIDTH $INSHAPE_HEIGHT
         --yolo_model $YOLO_TYPE
         --caffe_inference $NET_DEF"

if [ ! -z $GOLDEN ]; 
then
  echo -e "To check mAP score of the network please note the following"
  echo -e "   mAP score check for VOC and COCO data set is supported provided the data is in darknet style  "
  echo -e "   To get COCO data in darknet format run script https://github.com/pjreddie/darknet/blob/master/scripts/get_coco_dataset.sh  "
  echo -e "   To get VOC data in darknet format run script https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py  "
  echo -e "   All the images in the Val dataset should be provided in one folder and specified by --directory option"
  echo -e "   The corresponding groud truth label .txt files with same name as images should be provided in one folder and specified by --checkaccuracy option"  
  echo -e "   The script will generate the corresponding labels in ./out_labels folder "
  BASEOPT+=" --golden $GOLDEN"
  #BASEOPT+=" --directory $DIRECTORY"
  echo "Image Directory : $DIRECTORY"
  #DIRECTORY=/wrk/acceleration/shareData/COCO_Dataset/val2014_dummy/

fi

# Build options for appropriate python script
####################
# single image test
####################
if [ "$TEST" == "test_detect" ]; then
  TEST=yolo.py
  if [ -z ${DIRECTORY+x} ]; then
  DIRECTORY=${MLSUITE_ROOT}/apps/yolo/test_image_set/
  fi
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --dsp $DSP_WIDTH"
  BASEOPT+=" --net_def $NET_DEF"
  BASEOPT+=" --net_weights $NET_WEIGHTS"
  BASEOPT+=" --outsz $NUM_CLASSES"

elif [ "$TEST" == "darknet_detect" ]; then
  TEST=darknet_yolo.py
  if [ ! -z $GOLDEN ]; then
    #DIRECTORY=/opt/data/COCO_Dataset/val2014/
    DIRECTORY=/opt/data/deephi_data/coco_test_image/
  fi
  if [ -z ${DIRECTORY+x} ]; then
  DIRECTORY=${MLSUITE_ROOT}/apps/yolo/test_image_set/
  fi
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --dsp $DSP_WIDTH"
  BASEOPT+=" --net_def $NET_DEF"
  BASEOPT+=" --net_weights $NET_WEIGHTS"
  BASEOPT+=" --outsz $NUM_CLASSES"

# Build options for appropriate yolo C++ example
####################
# single image test
####################
elif [ "$TEST" == "yolo_cpp" ]; then
  cd yolo_cpp
  make
  if [ "$MODEL" == "yolo_v2_224" ]; then
  OUTSHAPE_WIDTH=7
  OUTSHAPE_HEIGHT=7
  OUTSHAPE_DEPTH=425
  elif [ "$MODEL" == "yolo_v2_416" ]; then
  OUTSHAPE_WIDTH=13
  OUTSHAPE_HEIGHT=13
  OUTSHAPE_DEPTH=425
  elif [ "$MODEL" == "yolo_v2_608" ]; then
  OUTSHAPE_WIDTH=19
  OUTSHAPE_HEIGHT=19
  OUTSHAPE_DEPTH=425
  fi
 
  DIRECTORY=${MLSUITE_ROOT}/apps/yolo/test_image_set/
  BASEOPT_CPP=" --xclbin $XCLBIN_PATH/$XCLBIN"
  BASEOPT_CPP+=" --netcfg $NETCFG" 
  BASEOPT_CPP+=" --weights $WEIGHTS"

  BASEOPT_CPP+=" --labels ${MLSUITE_ROOT}/apps/yolo/$LABELS"
  BASEOPT_CPP+=" --quantizecfg $QUANTCFG"
  BASEOPT_CPP+=" --img_input_scale $IMG_INPUT_SCALE"
  BASEOPT_CPP+=" --images $DIRECTORY"
  BASEOPT_CPP+=" --in_w $INSHAPE_WIDTH"
  BASEOPT_CPP+=" --in_h $INSHAPE_HEIGHT"
  BASEOPT_CPP+=" --in_d $INSHAPE_CHANNELS"
  BASEOPT_CPP+=" --out_w $OUTSHAPE_WIDTH"
  BASEOPT_CPP+=" --out_h $OUTSHAPE_HEIGHT"
  BASEOPT_CPP+=" --out_d $OUTSHAPE_DEPTH"
  BASEOPT_CPP+=" --batch_sz $BATCHSIZE"
  echo $BASEOPT_CPP
  ROOT_PATH=../../..
  OPENCV_LIB=${ROOT_PATH}/opencv_lib
  HDF5_PATH=${ROOT_PATH}/ext/hdf5
  NMS_LIB=${ROOT_PATH}/apps/yolo/nms
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_PATH/xfdnn/rt/xdnn_cpp/lib:$ROOT_PATH/ext/zmq/libs:$ROOT_PATH/ext/boost/libs:$ROOT_PATH/ext/sdx_build/runtime/lib/x86_64:${HDF5_PATH}/lib:${NMS_LIB}:$OPENCV_LIB
  cd -  
  
############################
# multi-process streaming 
############################  
elif [[ "$TEST" == "streaming_detect"* ]]; then
  
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=../../models/data/ilsvrc12/ilsvrc12_img_val
  fi

  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --numprepproc $NUMPREPPROC"    
  BASEOPT+=" --dsp $DSP_WIDTH"
  BASEOPT+=" --net_def $NET_DEF"
  BASEOPT+=" --net_weights $NET_WEIGHTS"
  BASEOPT+=" --outsz $NUM_CLASSES"

  if [ "$TEST" == "streaming_detect_benchmark" ]; then
    echo -e "run in benchkmark mode"
    BASEOPT+=" --benchmarkmode 1"
  else
    echo -e "not running in benchmark mode"
  fi 
  if [ "$PERPETUAL" == 1 ]; then 
    BASEOPT+=" --zmqpub --perpetual --deviceID $DEVICEID"
  fi

  TEST=mp_detect.py 
###########################
# multi-PE multi-network (Run two different networks simultaneously)
# runs with 8 bit quantization for now
###########################
elif [ "$TEST" == "multinet" ]; then
  TEST=test_classify_async_multinet.py
  if [ -z ${DIRECTORY+x} ]; then
    DIRECTORY=dog.jpg
  fi
  BASEOPT+=" --images $DIRECTORY"
  BASEOPT+=" --dsp $DSP_WIDTH --jsoncfg data/multinet.json"
else
  echo "Test was improperly specified..."
  exit 1
fi

if [ "$TEST" == "yolo_cpp" ]; then
  cd yolo_cpp
  echo "yolo v2 running on vcu1525 card"
  ./yolo.exe $BASEOPT_CPP
  cd -
elif [ "$MLSUITE_PLATFORM" == "gpu" ]; then
  QUANT_PTXT="./work/yolov2_quantized.prototxt"
  DIM_VAL="$INSHAPE_CHANNELS"
  DIM_VAL+=",$INSHAPE_WIDTH"
  DIM_VAL+=",$INSHAPE_HEIGHT"
  QUANTIZER_SCRIPT=${CAFFE_BACKEND_ROOT}/caffe/framework/python/quantize.py
  QUANT_OPTS=" --deploy_model $NET_DEF"
  QUANT_OPTS+=" --train_val_model $NET_DEF"
  QUANT_OPTS+=" --weights $NET_WEIGHTS"
  QUANT_OPTS+=" --quantized_deploy_model $QUANT_PTXT"
  QUANT_OPTS+=" --quantized_train_val_model work/yolov2_quantized_8Bit.prototxt" 
  QUANT_OPTS+=" --quantized_weights work/yolov2_without_bn_quantized.caffemodel"
  QUANT_OPTS+=" --calibration_directory ${CAFFE_BACKEND_ROOT}/caffe/framework/data/calibration_dataset_sample"
  QUANT_OPTS+=" --calibration_size 8"
  QUANT_OPTS+=" --bitwidths 8,8,8"
  QUANT_OPTS+=" --dims $DIM_VAL"
  QUANT_OPTS+=" --mean_value 0,0,0 --input_scale 0.003921568"
  #/usr/bin/python $QUANTIZER_SCRIPT $QUANT_OPTS
  NET_DEF=/wrk/acceleration/users/arun/MLsuite_fork/xilinx_yolov2_prelu.prototxt 
  NET_WEIGHTS=/wrk/acceleration/users/arun/MLsuite_fork/xilinx_yolov2_prelu.caffemodel
  DIRECTORY=/wrk/acceleration/shareData/deephi_data/coco_test_image/
  GPU_INFERENCE_OPTS=" --class_names_file $LABELS"
  GPU_INFERENCE_OPTS+=" --images $DIRECTORY"
  GPU_INFERENCE_OPTS+=" --deploy_model $NET_DEF"
  GPU_INFERENCE_OPTS+=" --weights $NET_WEIGHTS"
  GPU_INFERENCE_OPTS+=" --out_labels ./out_labels"  
  GPU_INFERENCE_OPTS+=" --dims  $INSHAPE_CHANNELS  $INSHAPE_WIDTH  $INSHAPE_HEIGHT "
  GPU_INFERENCE_OPTS+=" --mean_value 0 0 0"
  GPU_INFERENCE_OPTS+=" --pxscale 0.003921568"
  GPU_INFERENCE_OPTS+=" --backend_path ${CAFFE_BACKEND_ROOT}/caffe/framework/python"
   
  python yolo_gpu_inference.py $GPU_INFERENCE_OPTS
  
elif [ $ZELDA -eq "0" ]; then
  python $TEST $BASEOPT 2>&1 |tee single_img_out.txt
else
  gdb --args python $TEST $BASEOPT 
fi

if [ ! -z $GOLDEN ]; then
  python get_mAP_darknet.py --class_names_file $LABELS --ground_truth_labels  $GOLDEN  --detection_labels ./out_labels 2>&1 |tee batch_out.txt
fi


