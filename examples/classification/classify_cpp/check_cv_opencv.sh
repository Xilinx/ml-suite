#### checking g++ version and copy opencv libs from conda environments to MLsuite
## This is needed to building C++ APIs inference 
ROOT_PATH=${PWD}/../../..
CV_LIB_PATH=${ROOT_PATH}/opencv_lib
arg1=50400
GPPVERSION=$(g++ -dumpversion | sed -e 's/\.\([0-9][0-9]\)/\1/g' -e 's/\.\([0-9]\)/0\1/g' -e 's/^[0-9]\{3,4\}$/&00/')
#echo $GPPVERSION
#echo $arg1
if [ $arg1 -ge $GPPVERSION ]; then
   echo "g++ version matched"
else
   echo "g++ version >= 5.4.0 "
   exit
fi
DIRECTORY=$1
echo $DIRECTORY
echo " conda prefix"
if [ -d "$DIRECTORY" ]; then
   echo "conda evn path:"
   echo $DIRECTORY
   cp $DIRECTORY/lib/libopencv_core.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/libopencv_highgui.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/libopencv_imgproc.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/libopencv_imgcodecs.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/libz.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/libpng16.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/libjpeg.so* ${CV_LIB_PATH}
   cp $DIRECTORY/lib/liblzma.so* ${CV_LIB_PATH}
else
   echo "please run the conda enviroment "
   exit
fi
