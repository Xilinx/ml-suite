# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
# Mlsuite ROOT folder
ROOT_PATH :=${PWD}/../../..
BOOST_PATH := ${ROOT_PATH}/ext/boost
ZMQ_PATH := ${ROOT_PATH}/ext/zmq
OPENCV_LIB := /usr/lib/x86_64-linux-gnu
OPENCV_INC := /usr/include
XDNN_EXAMPLES := ${ROOT_PATH}/examples
XDNN_CPP_PATH := ${ROOT_PATH}/xfdnn/rt/xdnn_cpp
XDNN_CPP_LIB := ${ROOT_PATH}/xfdnn/rt/libs
HDF5_PATH=${ROOT_PATH}/ext/hdf5
OPENCV_SYM=-lopencv_core -lopencv_imgproc -lopencv_highgui 

CLASSIFY_SRC := \
        classify.cpp \
	    interface.cpp
classify : 
	@echo -e "\n[BUILD] Building Classification example \n\n"
	g++ -g -std=c++11 ${CLASSIFY_SRC} -I./ -I${XDNN_CPP_PATH} -I${OPENCV_INC} -I${BOOST_PATH} -I${ZMQ_PATH}/include -I${HDF5_PATH}/include -L/opt/OpenBLAS/lib/ -lopenblas -L${XDNN_CPP_LIB} -lxfdnn -L${ZMQ_PATH}/libs -lzmq -L${BOOST_PATH}/libs -lboost_thread -lboost_system -L/opt/xilinx/xrt/lib -lxilinxopencl -L${HDF5_PATH}/lib -lhdf5_cpp -L${OPENCV_LIB} ${OPENCV_SYM} -o classify.exe
	@echo -e "\n[BUILD] Done\n\n"

clean :
	rm *.exe
