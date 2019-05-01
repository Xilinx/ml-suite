ROOT_PATH=../../..
#OPENCV_LIB=/proj/sdxapps/users/kvraju/X_PULS_ML/mlsuite_v3/opencv_x86_64/lib
OPENCV_LIB=${ROOT_PATH}/
HDF5_PATH=${ROOT_PATH}/ext/hdf5
NMS_LIB=${ROOT_PATH}/apps/yolo/nms
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_PATH/xfdnn/rt/xdnn_cpp/lib:$ROOT_PATH/ext/zmq/libs:$ROOT_PATH/ext/boost/libs:$ROOT_PATH/ext/sdx_build/runtime/lib/x86_64:${HDF5_PATH}/lib:${NMS_LIB}:$OPENCV_LIB

source /opt/xilinx/xrt/setup.sh
echo "yolo v2 running on vcu1525 card"

#./yolo.exe --xclbin=$ROOT_PATH/overlaybins/1525/overlay_2.xclbin --data_dir=$ROOT_PATH/apps/yolo/work_224/yolov2.caffemodel_data --cmd_json=$ROOT_PATH/apps/yolo/work_224/yolo.cmds.json --quant_json=$ROOT_PATH/apps/yolo/work_224/yolo_deploy_224_8b.json --labelfile=$ROOT_PATH/examples/deployment_modes/coco_names.txt  --in_img=$ROOT_PATH/xfdnn/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg --in_w=224 --in_h=224 --out_w=7 --out_h=7 --out_d=425

#./yolo.exe --xclbin=$ROOT_PATH/overlaybins/1525/overlay_2.xclbin --data_dir=$ROOT_PATH/apps/yolo/work/yolov2.caffemodel_data --cmd_json=$ROOT_PATH/apps/yolo/work_416/yolo.cmds.json --quant_json=$ROOT_PATH/apps/yolo/work_416/yolo_deploy_416_8b.json --labelfile=$ROOT_PATH/examples/deployment_modes/coco_names.txt --in_img=$ROOT_PATH/xfdnn/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg --in_w=416 --in_h=416 --out_w=13 --out_h=13 --out_d=425 

#./yolo.exe --xclbin=$ROOT_PATH/overlaybins/1525/overlay_2.xclbin --data_dir=$ROOT_PATH/apps/yolo/work/yolov2.caffemodel_data --cmd_json=$ROOT_PATH/apps/yolo/work/yolo.cmds.json --quant_json=$ROOT_PATH/apps/yolo/work/yolo_deploy_608_8b.json --labelfile=$ROOT_PATH/examples/deployment_modes/coco_names.txt --in_img=$ROOT_PATH/xfdnn/tools/quantize/calibration_directory/4788821373_441cd29c9f_z.jpg --in_w=608 --in_h=608 --out_w=19 --out_h=19 --out_d=425

./yolo.exe --xclbin $ROOT_PATH/overlaybins/1525/overlay_2.xclbin --datadir $ROOT_PATH/apps/yolo/work/yolov2.caffemodel_data --netcfg $ROOT_PATH/apps/yolo/work/yolo.cmds.json --quantizecfg $ROOT_PATH/apps/yolo/work/yolo_deploy_608_8b.json --labels $ROOT_PATH/examples/deployment_modes/coco_names.txt --images $ROOT_PATH/xfdnn/tools/quantize/calibration_directory --in_w 608 --in_h 608 --out_w 19 --out_h 19 --out_d 425 --batch_sz 2
