# Mlsuite ROOT folder
ROOT_PATH :=${PWD}/../../..
BOOST_PATH := ${ROOT_PATH}/ext/boost
ZMQ_PATH := ${ROOT_PATH}/ext/zmq
OPENCV_DIR := ${ROOT_PATH}/opencv_lib
XILINX_SDACCEL :=${ROOT_PATH}/ext/sdx_build
XDNN_EXAMPLES := ${ROOT_PATH}/examples
XDNN_CPP_PATH := ${ROOT_PATH}/xfdnn/rt/xdnn_cpp
CBLAS_PATH := ${ROOT_PATH}/ext/OpenBLAS
HDF5_PATH=${ROOT_PATH}/ext/hdf5
OPENCV_SYM=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs -lpthread -llzma -lpng16 -ljpeg -ltiff -lz
RESNET50_SRC := \
        classify.cpp 
resnet : do_script
	@echo -e "\n[BUILD] Building Classification example \n\n"
	g++ -g -std=c++11 ${RESNET50_SRC} -I./ -I${XDNN_CPP_PATH} -I${CONDA_PREFIX}/include -I${BOOST_PATH} -I${ZMQ_PATH}/include -I${CBLAS_PATH}/include -I${HDF5_PATH}/include ${ROOT_PATH}/ext/OpenBLAS/libopenblas.a -L${OPENCV_DIR}/ ${OPENCV_SYM} -L${XDNN_CPP_PATH}/lib -lxfdnn -L${ZMQ_PATH}/libs -lzmq -L${BOOST_PATH}/lib -lboost_thread -lboost_system -L${XILINX_SDACCEL}/runtime/lib/x86_64 -lxilinxopencl -L${HDF5_PATH}/lib -lhdf5_cpp -o ${XDNN_EXAMPLES}/deployment_modes/classify_cpp/classify.exe
	@echo -e "\n[BUILD] Done\n\n"

.PHONY: do_script

do_script:
	echo ${CONDA__PREFIX}
	mkdir -p ${ROOT_PATH}/opencv_lib
	./check_cv_opencv.sh ${CONDA_PREFIX}

clean :
	rm *.exe
