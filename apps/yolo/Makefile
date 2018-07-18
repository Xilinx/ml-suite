## This Makefile will build libxfdnn.so

## At this time, Users will not need this, as the repo will ship with prebuilt binaries

all:
	cd ../../xfdnn/rt/xdnn_cpp && $(MAKE) -j 10 XBLASOPTS="-O3 -D CBLAS -D XDNN_V2"
	#cd ../../xfdnn/rt/xdnn_cpp && $(MAKE) -j 10 XBLASOPTS="-O3 -D XDNN_V2 -D XDNN_V2_XDNN_FORMAT -g"

clean:
	cd ../../xfdnn/rt/xdnn_cpp && $(MAKE) clean

