FROM ubuntu:16.04

ENV DEBIAN_FRONTEND noninteractive

# Libraries
RUN apt-get update && apt-get install -y --no-install-recommends\
 build-essential\
 ca-certificates\
 cmake\
 rsync\
 sudo\
 curl\
 wget\
 unzip\
 git\
 vim\
 graphviz\
 python-dev\
 python-skimage\
 libatlas-base-dev\
 libboost-all-dev\
 libgflags-dev\
 libgoogle-glog-dev\
 libhdf5-serial-dev\
 libleveldb-dev\
 liblmdb-dev\
 libopencv-dev\
 libprotobuf-dev\
 libsnappy-dev\
 protobuf-compiler\
 libyaml-cpp-dev

# Install XRT
RUN wget https://www.xilinx.com/support/download/xrt/xrt-201802-2-1-83/xrt_201802.2.1.83_16.04.deb && \
 sudo apt install ./*.deb -y && rm -rf ./*.deb && rm -rf /var/lib/apt/lists/*

# Get pip setuptools and wheel
RUN cd /tmp && curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && python get-pip.py && rm get-pip.py

# Python Module Dependencies
RUN pip --no-cache-dir install\
 numpy==1.14.3\
 six==1.11.0\
 ck\
 pydot\
 pydot-ng\
 graphviz\
 h5py\
 zmq\
 protobuf\
 opencv-python\
 pytest\
 pillow\
 Flask\
 jupyter\
 ipykernel\
 matplotlib\
 enum34

# Set up non-root user for file and process ownership
RUN groupadd users -g 100 -f && useradd -g users -ms /bin/bash mluser -u 1185 && usermod -aG sudo mluser && passwd -d mluser && \
 mkdir /scratch && chmod 777 /scratch && chmod 777 /opt && chmod 777 /tmp
USER mluser

# OpenBLAS Setup To /opt/OpenBLAS
ENV LD_LIBRARY_PATH=/opt/OpenBLAS/lib
ENV OPENBLAS=/opt/OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git /scratch/OpenBLAS && \
 cd /scratch/OpenBLAS && make -j NO_LAPACK=1 NO_LAPACKE=1 CC=/usr/bin/gcc USE_THREAD=1 && \
 make PREFIX=$OPENBLAS install && rm -rf /scratch/OpenBLAS && \
 cd $OPENBLAS && find . -name "*.a" -exec rm {} \; && rm -rf include 
# May need to disable AVX2 for old machines
# NO_AVX2=1

# Caffe Decent Setup To /opt/caffe
ENV CAFFE_ROOT=/opt/caffe
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
ADD caffe_decent.tar.gz /opt/

# Public ml-suite Setup To /opt/ml-suite switch to latest branch
ENV MLSUITE_ROOT=/opt/ml-suite
RUN git clone https://github.com/Xilinx/ml-suite.git $MLSUITE_ROOT && \
 cd $MLSUITE_ROOT && git checkout latest && git pull && rm -rf .git* && rm -rf docker

# Models Directory (Models to be pulled after container start)
RUN mkdir -p /opt/models/caffe

# Automatically Set Environment Variables
RUN echo "export PATH=$PATH:~/.local/bin" >> /home/mluser/.bashrc && \
 echo ". /opt/ml-suite/overlaybins/setup.sh" >> /home/mluser/.bashrc
