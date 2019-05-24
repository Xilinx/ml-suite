# Select platform
#export PLATFORM=alveo-u250
#export PLATFORM=alveo-u200-ml
#export PLATFORM=alveo-u200
export PLATFORM=1525

# Export MLSuite path
#export MLSUITE_ROOT=..
MODELS_DIR_PATH=/wrk/acceleration/models/caffe
PRUNED_NW_PATH=/wrk/acceleration/models/caffe/pruned_nw
export MLSUITE_ROOT=/wrk/acceleration/MLsuite_Embedded/anup/deephi_quant/MLsuite

# Enable for crontab runs
source /wrk/acceleration/MLsuite_Embedded/anup/virenv/anaconda2/bin/activate ml-suite

# pull latest 
git pull -r

# pull lfs files
export PATH=$PATH:/wrk/acceleration/MLsuite_Embedded/anup/gitlfs/
git lfs pull

# Enable below to build xdnn lib on CentOS 7.4
#export PATH=/tools/batonroot/rodin/devkits/lnx64/binutils-2.26/bin:/tools/batonroot/rodin/devkits/lnx64/make-4.1/bin:/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/bin:$PATH
#export LD_LIBRARY_PATH=/tools/batonroot/rodin/devkits/lnx64/gcc-7.1.0/lib64:$LD_LIBRARY_PATH

# Build rt
cd ../xfdnn/rt/xdnn_cpp
make clean;make -j8
cd -

rm nw_status.txt

# Adding data to log file
echo "######## Status of Deephi networks ########" >> nw_status.txt
echo "Platform : $PLATFORM" >> nw_status.txt
gitid=$(git log --format="%H" -n 1)
#date=$(date)
echo "Git commit ID : $gitid" >> nw_status.txt
echo "Date : $(date)" >> nw_status.txt
printf "\n\n" >> nw_status.txt

# Logs directory
rm -r output_logs
mkdir output_logs

# Disable if dont want to run any of the networks
deephi_nw_run=1
pruned_nw_run=1

#for i in "nooptim" "standard" "mix"
#for i in "mix1"
#for i in "mix" "mix1"
#for i in "standard" "mix" "mix1"
for i in "mix1" "default"
do

    if [ $deephi_nw_run -eq 1 ];
    then
    
        echo "#####  network List #####" >> nw_status.txt
             
        ./run_network.sh $MODELS_DIR_PATH/bvlc_googlenet_without_lrn bvlc_googlenet_without_lrn_deploy.prototxt bvlc_googlenet_without_lrn.caffemodel 104 117 123 1.0 $i bvlc_googlenet_without_lrn

        #exit 1;
       
        ./run_network.sh $MODELS_DIR_PATH/inception_v2/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i inception_v2
        
        ./run_network.sh $MODELS_DIR_PATH/inception_v3 inception_v3_without_bn_deploy.prototxt inception_v3_without_bn.caffemodel 104 117 123 1.0 $i inception_v3
        
        ./run_network.sh $MODELS_DIR_PATH/inception_v4/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i inception_v4
        
        ./run_network.sh $MODELS_DIR_PATH/resnet50 resnet50_without_bn_deploy.prototxt resnet50_without_bn.caffemodel 104 117 123 1.0 $i resnet50_v1
        
        ./run_network.sh $MODELS_DIR_PATH/resnet50_v2 deploy.prototxt deploy.caffemodel 0 0 0 1.0 $i resnet50_v2
        
        ./run_network.sh $MODELS_DIR_PATH/squeezenet squeezenet_deploy.prototxt squeezenet.caffemodel 104 117 123 1.0 $i squeezenet
        
        ./run_network.sh $MODELS_DIR_PATH/vgg16 vgg16_deploy.prototxt vgg16.caffemodel 104 117 123 1.0 $i vgg16
    
        #exit 1; 
    fi
    
    
    if [ $pruned_nw_run -eq 1 ];
    then    
        echo "#####  Pruned network List #####" >> nw_status.txt
    
        ./run_network.sh $PRUNED_NW_PATH/Inception_v1_prune_2.27G/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i Inception_v1_prune_2.27G

        #exit 1;
        
        ./run_network.sh $PRUNED_NW_PATH/Inception_v1/prune_1.8G/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i inception_v1_prune_1.8G
        
        ./run_network.sh $PRUNED_NW_PATH/SqueezeNet1.1/prune_round1_691M/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i SqueezeNet1.1_prune_round1_691M
        
        ./run_network.sh $PRUNED_NW_PATH/resnet_50_v1/prune_round10_2.6G/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i resnet_50_v1_prune_round10_2.6G
        
        ./run_network.sh $PRUNED_NW_PATH/resnet_50_v1/prune_round5_3.7G/quantize_results deploy.prototxt deploy.caffemodel 104 117 123 1.0 $i resnet_50_v1_prune_round5_3.7G
    fi

done


################## Yolo call

NW_LOG_DIR=output_logs/yolov2
NW_NAME=yolov2

mkdir $NW_LOG_DIR

YOLO_RUN_DIR=$MLSUITE_ROOT/apps/yolo

# Goto yolo directory
cd $YOLO_RUN_DIR
./run.sh -p $PLATFORM -t test_detect -m yolo_v2_608 -k v3 -b 8 -g /wrk/acceleration/shareData/COCO_Dataset/labels/val2014 -d /wrk/acceleration/shareData/COCO_Dataset/val2014_dummy

cd -

cp $YOLO_RUN_DIR/single_img_out.txt $NW_LOG_DIR
cp $YOLO_RUN_DIR/batch_out.txt $NW_LOG_DIR

printf "\n\n" >> nw_status.txt

echo "*** Network : $NW_NAME" >> nw_status.txt
printf "\n" >> nw_status.txt
echo "Run Directory Path        : $MLSUITE_ROOT/apps/yolo" >> nw_status.txt
echo "Output Log Directory Path : $(pwd)/$NW_LOG_DIR" >> nw_status.txt

echo "compile mode : default" >> nw_status.txt
grep "hw_counter" $NW_LOG_DIR/single_img_out.txt | tail -1 >> nw_status.txt
grep "exec_xdnn" $NW_LOG_DIR/single_img_out.txt | tail -1 >> nw_status.txt

grep "mAP" $NW_LOG_DIR/batch_out.txt >> nw_status.txt


printf "\n" >> nw_status.txt


################## Table Generation

python gen_table.py nw_status.txt

cur_time=$(date +"%d%b%y_%H-%M")
cur_date=$(date +"%d%b%y")
echo "$cur_time"
logs_dir=track_logs/$cur_time
echo "$logs_dir"
mkdir $logs_dir

cp nw_status.txt $logs_dir/
cp xfdnn_nightly.csv $logs_dir/ 

mv nw_status.txt output_logs/nw_status_$cur_date.txt
mv xfdnn_nightly.csv output_logs/xfdnn_nightly_$cur_date.csv

#echo -e "Please find attached log for accuracy details and latency numbers. \nPlease find confluence page link for details about how to run the scripts - http://confluence.xilinx.com/display/XSW/Run+scripts+for+Deephi+networks" | mailx -a output_logs/nw_status_$cur_date.txt -a output_logs/xfdnn_nightly_$cur_date.csv -s "Deephi NW Accuracy" anup@xilinx.com
#asirasa@xilinx.com aaronn-all@xilinx.com elliott-all@xilinx.com sumitn-all@xilinx.com

#echo "Please find attached log for accuracy details and latency numbers. Added 8 networks in the list and pruned networks too. Need to add Yolov2, mask-RCNN and SSD." | mailx -a nw_status.txt -s "Deephi NW Accuracy" anup@xilinx.com sumitn@xilinx.com asirasa@xilinx.com
