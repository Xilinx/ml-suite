ROOT_PATH=../../..
#OPENCV_LIB=/proj/sdxapps/users/kvraju/X_PULS_ML/mlsuite_v3/opencv_x86_64/lib
OPENCV_LIB=${ROOT_PATH}/opencv_lib
HDF5_PATH=${ROOT_PATH}/ext/hdf5
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROOT_PATH/xfdnn/rt/xdnn_cpp/lib:$ROOT_PATH/ext/zmq/libs:$ROOT_PATH/ext/boost/libs:$ROOT_PATH/ext/sdx_build/runtime/lib/x86_64:${HDF5_PATH}/lib:$OPENCV_LIB

source /opt/xilinx/xrt/setup.sh
echo "Classification with xdnn_v2 running on vcu1525 card"

#./resnet50.exe --xclbin $ROOT_PATH/overlaybins/1525/overlay_2.xclbin --datadir $ROOT_PATH/examples/deployment_modes/data/resnet50_data --netcfg $ROOT_PATH/examples/deployment_modes/data/resnet50_56.json --quantizecfg $ROOT_PATH/examples/deployment_modes/data/resnet50_8b.json --labels $ROOT_PATH/examples/deployment_modes/synset_words.txt --image $ROOT_PATH/examples/deployment_modes/dog.jpg --in_w 224 --in_h 224 --out_w 1 --out_h 1 --out_d 1024 --batch_sz 2

#./classify.exe --xclbin $ROOT_PATH/overlaybins/1525/overlay_2.xclbin --datadir $ROOT_PATH/examples/deployment_modes/data/resnet50_data --netcfg $ROOT_PATH/examples/deployment_modes/data/resnet50_56.json --quantizecfg $ROOT_PATH/examples/deployment_modes/data/resnet50_8b.json --labels $ROOT_PATH/examples/deployment_modes/synset_words.txt --image $ROOT_PATH/examples/deployment_modes/dog.jpg --in_w 224 --in_h 224 --out_w 1 --out_h 1 --out_d 2048 --batch_sz 2

./classify.exe --xclbin $ROOT_PATH/overlaybins/1525/overlay_2.xclbin --datadir $ROOT_PATH/examples/deployment_modes/data/googlenet_v1_data --netcfg $ROOT_PATH/examples/deployment_modes/data/googlenet_v1_56.json --quantizecfg $ROOT_PATH/examples/deployment_modes/data/googlenet_v1_8b.json --labels $ROOT_PATH/examples/deployment_modes/synset_words.txt --image $ROOT_PATH/examples/deployment_modes/dog.jpg --in_w 224 --in_h 224 --out_w 1 --out_h 1 --out_d 1024 --batch_sz 2
