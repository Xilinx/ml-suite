gcc -c -fPIC nms.c -o nms.o
#g++ -c -fPIC nms.c -o nms.o
g++ -shared -Wl,-soname,libnms.so -o libnms.so nms.o
