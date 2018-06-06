# Usage: 
# make -j 10 XBLASOPTS="-D ZMQ -D CBLAS" 
# LIBRARY_PATH=/usr/lib/x86_64-linux-gnu make -j 10 CXX="/tools/batonroot/rodin/devkits/lnx64/gcc-4.8.0/bin/g++" XBLASOPTS="-O3"

XBLAS_SRC := \
	xblas.cpp \
	xdnn.cpp \
	xmlrt.cpp

XBLAS_LIB=libxblas.so
XFDNN_LIB=libxfdnn.so

BOOST_SRC=../../../ext/boost/src
BOOST_LIB=../../../ext/boost/lib
XILINX_SDACCEL := ../../../ext/sdx_build

XBLASOPTS := $(XBLASOPTS)
XDNNOPTS := $(XDNNOPTS)

INCLUDE := -I$(XILINX_SDACCEL)/include -I$(XILINX_SDACCEL)/runtime/include/1_2 
ifneq (,$(findstring ZMQ,$(XBLASOPTS)))
INCLUDE += -I../../../ext/zmq/include 
endif

ifneq (,$(findstring MKL,$(XBLASOPTS)))
INCLUDE += -I../../../ext/intel/mkl/include
endif
ifneq (,$(findstring CBLAS,$(XBLASOPTS)))
INCLUDE += -I./blas/include 
endif

CCOPTS := -D __USE_XOPEN2K8 $(XBLASOPTS) $(XDNNOPTS) 
CXX := g++
CBLAS_LIB_OPTION :=

ifneq (,$(findstring SANITIZE,$(XBLASOPTS)))
#CBLAS_LIB_OPTION += -lubsan -lasan  -fsanitize-recover=address
#CCOPTS += -fno-omit-frame-pointer -lubsan -lasan -fuse-ld=gold -fsanitize=address,undefined,bounds-strict  -fsanitize-recover=address
CBLAS_LIB_OPTION += -lubsan -lasan
CCOPTS += -fno-omit-frame-pointer -lubsan -lasan -fuse-ld=gold -fsanitize=undefined,bounds-strict
CXX := g++-6
endif

ifneq (,$(findstring MKL,$(XBLASOPTS)))
CBLAS_LIB_PATH := ../../../ext/intel/mkl/lib/intel64
CBLAS_LIB := $(CBLAS_LIB_PATH)/libmkl_rt.so
CBLAS_LIB_OPTION += -L$(CBLAS_LIB_PATH) -lmkl_rt
endif
ifneq (,$(findstring CBLAS,$(XBLASOPTS)))
CBLAS_LIB := ./blas/libopenblas.a
CBLAS_LIB_OPTION += $(CBLAS_LIB)
endif

ifneq (,$(findstring ZMQ,$(XBLASOPTS)))
ZMQ_LIB := ../../../ext/zmq/libs/libzmq.so
CBLAS_LIB_OPTION += $(CBLAS_LIB_OPTION) -lzmq -L../../../ext/zmq/libs
endif

XBLAS_OBJS := $(addprefix objs/,$(addsuffix .o,$(basename $(XBLAS_SRC))))

.PHONY: all

objs/%.o: %.cpp
	@mkdir -p objs/$(dir $<)
	$(CXX) $(CCOPTS) -I $(BOOST_SRC) -L $(BOOST_LIB) -D XBLAS -fPIC -std=c++11 -lboost_thread -c -o $@ $(INCLUDE) $<


all: $(XBLAS_LIB) $(XFDNN_LIB)
	@echo Done

$(XFDNN_LIB): $(XBLAS_LIB)
	cp lib/$(XBLAS_LIB) lib/$(XFDNN_LIB)

$(XBLAS_LIB): $(XBLAS_OBJS) $(ZMQ_LIB) 
	mkdir -p lib
	$(CXX) -shared -L $(BOOST_LIB) -L$(XILINX_SDACCEL)/runtime/lib/x86_64 -pthread -o lib/$@ $(XBLAS_OBJS) $(CBLAS_LIB_OPTION) $(ZMQ_LIB) $(CC_OPT) -lboost_thread -lxilinxopencl
	chmod a+rx lib
	chmod a+r lib/*

clean:
	rm -rf objs lib
