#!/bin/bash
echo "### Cleaning Stale Files From Previous Run ###"
#rm -rf output_logs
rm nw_status.txt

# Logs directory
mkdir output_logs

# Select platform
if [ -z "$MLSUITE_PLATFORM" ]; then
  export MLSUITE_PLATFORM="(unknown)"
  AUTODETECT_PLATFORM=`python -c "from xfdnn.rt.xdnn import getHostDeviceName; print(getHostDeviceName(None).decode('utf-8'))" | tr -d '\n'`
  if [ ! -z "$AUTODETECT_PLATFORM" -a $? -eq 0 -a `echo "$AUTODETECT_PLATFORM" | wc -w` -eq "1" ]; then
      #echo "Auto-detected platform: ${AUTODETECT_PLATFORM}"
      export MLSUITE_PLATFORM=${AUTODETECT_PLATFORM}
  else
      echo "Warning: failed to auto-detect platform. Please manually specify platform with -p"
  fi
else
  export MLSUITE_PLATFORM=$MLSUITE_PLATFORM
fi


export PLATFORM=$MLSUITE_PLATFORM

# Export MLSuite path
export MLSUITE_ROOT=/opt/ml-suite
export XFDNN_ROOT=/opt/ml-suite
export SSD_DATA=/opt/ml-suite/share/data

# pull latest 
#git pull -r

# pull lfs files
#export PATH=$PATH:/wrk/acceleration/MLsuite_Embedded/anup/gitlfs/
#git lfs pull

# Enable below to build xdnn lib on CentOS 7.4
#export PATH=/tools/batonroot/rodin/devkits/lnx64/binutils-2.26/bin:/tools/batonroot/rodin/devkits/lnx64/make-4.1/bin:/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/bin:$PATH
#export LD_LIBRARY_PATH=/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/lib64:$LD_LIBRARY_PATH

# Build rt
#cd ../xfdnn/rt/xdnn_cpp
#make clean;
#make -j8
#cd -

#Build nms for yolo
#cd $MLSUITE_ROOT/apps/yolo/nms
#make
#cd -

# Run docker
#./$MLSUITE_ROOT/anup_docker/docker_run.sh ubuntu-16.04-caffe-mls-1.4
#cd $MLSUITE_ROOT/test_deephi

# Set the model dir base path
MODELS_DIR_PATH=/opt/ml-suite/share/models/caffe
#MODELS_DIR_PATH=/opt/ml-suite/models/container/caffe/
#MODELS_DIR_PATH=/opt/models/caffe/deephi_nw_anup

# Adding data to log file
echo "######## Status of Deephi networks ########" >> nw_status.txt
echo "Platform : $PLATFORM" >> nw_status.txt
gitid=$(git log --format="%H" -n 1)
#date=$(date)
echo "Git commit ID : $gitid" >> nw_status.txt
echo "Date : $(date)" >> nw_status.txt
printf "\n\n" >> nw_status.txt

# Disable if dont want to run any of the networks
deephi_nw_run=1
pruned_nw_run=0
yolo_nw_run=0
yolo_prelu_nw_run=0

#for i in "mix1" "default"
for i in "default" 
do

    if [ $deephi_nw_run -eq 1 ];
    then
        echo "#####  network List #####" >> nw_status.txt
        #./run_network.sh $MODELS_DIR_PATH/bvlc_googlenet $i
        #./run_network.sh $MODELS_DIR_PATH/inception_v1 $i
        #./run_network.sh $MODELS_DIR_PATH/inception_v2 $i
        #./run_network.sh $MODELS_DIR_PATH/inception_v3 $i
        #./run_network.sh $MODELS_DIR_PATH/inception_v4 $i
        #./run_network.sh $MODELS_DIR_PATH/resnet50_v1 $i
        #./run_network.sh $MODELS_DIR_PATH/resnet50_v2 $i
        #./run_network.sh $MODELS_DIR_PATH/squeezenet $i
        #./run_network.sh $MODELS_DIR_PATH/vgg16 $i
        ./run_network.sh $MODELS_DIR_PATH/inception_v2_ssd $i
    
        #exit 1; 
    fi
    
    
    if [ $pruned_nw_run -eq 1 ];
    then    
        echo "#####  Pruned network List #####" >> nw_status.txt
    
        #./run_network.sh $MODELS_DIR_PATH/Inception_v1_prune_2.27G $i
        ./run_network.sh $MODELS_DIR_PATH/resnet_50_v1_prune_round10_2.6G $i
        ./run_network.sh $MODELS_DIR_PATH/resnet_50_v1_prune_round5_3.7G $i
    fi

done


if [ $yolo_nw_run -eq 1 ];
then    
################## Yolo standard network call

rm ${MLSUITE_ROOT}/apps/yolo/single_img_out.txt
rm ${MLSUITE_ROOT}/apps/yolo/batch_out.txt

NW_LOG_DIR=output_logs/yolov2_standard
NW_NAME=yolov2_standard

mkdir $NW_LOG_DIR

YOLO_RUN_DIR=$MLSUITE_ROOT/apps/yolo

printf "\n\n" >> nw_status.txt
    
echo "*** Network : $NW_NAME" >> nw_status.txt
printf "\n" >> nw_status.txt
echo "Run Directory Path        : $MLSUITE_ROOT/apps/yolo" >> nw_status.txt
echo "Output Log Directory Path : $(pwd)/$NW_LOG_DIR" >> nw_status.txt
echo "compile mode : default" >> nw_status.txt
 
for run_mode in "latency" "throughput"
do
    # Goto yolo directory
    cd $YOLO_RUN_DIR
    echo "$YOLO_RUN_DIR"
    
    ./run.sh -p $PLATFORM -t test_detect -m yolo_v2_608 -k v3 -b 8 -g /opt/data/COCO_Dataset/labels/val2014 -d /opt/data/COCO_Dataset/val2014_dummy -compilerOpt $run_mode
    #./run.sh -p $PLATFORM -t test_detect -m yolo_v2_standard_416 -k v3 -b 8 -g /opt/data/COCO_Dataset/labels/val2014 -d /opt/data/COCO_Dataset/val2014_dummy -compilerOpt $run_mode
    
    cd -
    
    cp $YOLO_RUN_DIR/single_img_out.txt ${NW_LOG_DIR}/${run_mode}_single_img_out.txt
    mkdir ${NW_LOG_DIR}/${run_mode}
    cp $YOLO_RUN_DIR/work/* ${NW_LOG_DIR}/${run_mode}/
    cp $YOLO_RUN_DIR/batch_out.txt ${NW_LOG_DIR}/${run_mode}_batch_out.txt
   
    echo "Run mode : ${run_mode}" >> nw_status.txt
    grep "hw_counter" ${NW_LOG_DIR}/${run_mode}_single_img_out.txt | tail -1 >> nw_status.txt
    grep "exec_xdnn" ${NW_LOG_DIR}/${run_mode}_single_img_out.txt | tail -1 >> nw_status.txt
    grep "mAP" ${NW_LOG_DIR}/${run_mode}_batch_out.txt >> nw_status.txt

done
echo "*** Network End" >> nw_status.txt
fi


if [ $yolo_prelu_nw_run -eq 1 ];
then    
################## Yolo prelu network call

rm ${MLSUITE_ROOT}/apps/yolo/single_img_out.txt
rm ${MLSUITE_ROOT}/apps/yolo/batch_out.txt

NW_LOG_DIR=output_logs/yolov2_prelu
NW_NAME=yolov2_prelu

mkdir $NW_LOG_DIR

YOLO_RUN_DIR=$MLSUITE_ROOT/apps/yolo

printf "\n\n" >> nw_status.txt
    
echo "*** Network : $NW_NAME" >> nw_status.txt
printf "\n" >> nw_status.txt
echo "Run Directory Path        : $MLSUITE_ROOT/apps/yolo" >> nw_status.txt
echo "Output Log Directory Path : $(pwd)/$NW_LOG_DIR" >> nw_status.txt
echo "compile mode : default" >> nw_status.txt
 
for run_mode in "latency" "throughput"
do

    # Goto yolo directory
    cd $YOLO_RUN_DIR

    ./run.sh -p $PLATFORM -t test_detect -m yolo_v2_prelu_608 -k v3 -b 8 -g /opt/data/COCO_Dataset/labels/val2014 -d /opt/data/COCO_Dataset/val2014_dummy -compilerOpt $run_mode
    
    cd -
    
    cp $YOLO_RUN_DIR/single_img_out.txt ${NW_LOG_DIR}/${run_mode}_single_img_out.txt
    mkdir ${NW_LOG_DIR}/${run_mode}
    cp $YOLO_RUN_DIR/work/* ${NW_LOG_DIR}/${run_mode}/
    cp $YOLO_RUN_DIR/batch_out.txt ${NW_LOG_DIR}/${run_mode}_batch_out.txt
   
    echo "Run mode : ${run_mode}" >> nw_status.txt
    grep "hw_counter" ${NW_LOG_DIR}/${run_mode}_single_img_out.txt | tail -1 >> nw_status.txt
    grep "exec_xdnn" ${NW_LOG_DIR}/${run_mode}_single_img_out.txt | tail -1 >> nw_status.txt
    grep "mAP" ${NW_LOG_DIR}/${run_mode}_batch_out.txt >> nw_status.txt
    
done
echo "*** Network End" >> nw_status.txt
fi

################## Table Generation

python gen_table.py nw_status.txt

cur_time=$(date +"%d%b%y_%H-%M")
cur_date=$(date +"%d%b%y")
echo "$cur_time"

mv nw_status.txt output_logs/nw_status_$cur_date.txt
mv xfdnn_nightly.csv output_logs/xfdnn_nightly_$cur_date.csv
#mv output_logs output_logs_${cur_date}

