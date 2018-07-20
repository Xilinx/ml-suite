current_dir := $(shell pwd)
BOOST_HOME=../ext/boost
BOOST_LIB=.${BOOST_HOME}/libs
GEMX_SRC := ./src/gemx_host_c_api.cpp
OPENCL_LIB=${current_dir}/runtime/lib/x86_64
OPENCL_INC=${current_dir}/runtime/include/1_2 
GEMX_OUT := libgemxhost.so

CXX := g++

GEMX_OBJS := $(addprefix objs/,$(addsuffix .o,$(basename $(GEMX_SRC))))

GEMX_INCLUDE := -I $(BOOST_HOME) -I ./src -I$(OPENCL_INC) 
GEMX_DEF = -DBOOST_COMPUTE_HAVE_THREAD_LOCAL -DCL_USE_DEPRECATED_OPENCL_1_1_APIS -DBOOST_COMPUTE_THREAD_SAFE
GEMX_LIBDIR = -L$(BOOST_LIB) -L$(OPENCL_LIB)
GEMX_CXXFLAGS = -O3 -std=c++14 -fPIC -Wextra -Wall -Wno-ignored-attributes -Wno-unused-parameter -Wno-unused-variable
GEMX_LDFLAGS = -fPIC -Wl,-rpath,$(OPENCL_LIB),-rpath,$(BOOST_LIB)
              
GEMX_LIB = -lxilinxopencl -lz -lstdc++ -lrt -pthread 
                             
.PHONY: all

objs/%.o: %.cpp
	@mkdir -p objs/$(dir $<)
	$(CXX) -shared $(GEMX_INCLUDE) $(GEMX_DEF) $(GEMX_CXXFLAGS) -c -o $@ $<

all: $(GEMX_OUT)
	@echo Done

$(GEMX_OUT): $(GEMX_OBJS) 
	mkdir -p lib
	$(CXX) -shared -o lib/$@ $< $(GEMX_LIBDIR) $(GEMX_LDFLAGS) $(GEMX_LIB) 
	chmod a+rx lib
	chmod a+r lib/*

clean:
	rm -rf objs lib
	

