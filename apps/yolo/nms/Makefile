#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
NMSPATH=./nms_20180209

all: NMS libnms.so

NMS:
	cd ${NMSPATH} && $(MAKE)

libnms.so: nms.o
	g++ -shared -Wl,-soname,libnms.so -o libnms.so nms.o ${NMSPATH}/build/libnms.a

nms.o: nms.c
	gcc -g -c -fPIC -I${NMSPATH}/include nms.c -o nms.o

clean:
	rm -rf *.o *.so *.pyc
