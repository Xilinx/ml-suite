ROOT_PATH=../../..
OPENCV_LIB=/usr/lib/x86_64-linux-gnu
HDF5_PATH=${ROOT_PATH}/ext/hdf5
NMS_LIB=${ROOT_PATH}/apps/yolo/nms
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_PATH/xfdnn/rt/libs:$ROOT_PATH/ext/zmq/libs:$ROOT_PATH/ext/boost/libs:$ROOT_PATH/ext/sdx_build/runtime/lib/x86_64:${HDF5_PATH}/lib:${NMS_LIB}:$OPENCV_LIB

source /opt/xilinx/xrt/setup.sh
echo "yolo v2 running on vcu1525 card"


./yolo.exe --xclbin $ROOT_PATH/overlaybins/1525/overlay_2.xclbin --datadir $ROOT_PATH/apps/yolo/work/yolov2.caffemodel_data --netcfg $ROOT_PATH/apps/yolo/work/yolo.cmds.json --quantizecfg $ROOT_PATH/apps/yolo/work/yolo_deploy_608_8b.json --labels $ROOT_PATH/examples/deployment_modes/coco_names.txt --images $ROOT_PATH/xfdnn/tools/quantize/calibration_directory --in_w 608 --in_h 608 --out_w 19 --out_h 19 --out_d 425 --batch_sz 2
