
# If you call your environment something else, need to edit the below line
cd ~/anaconda2/envs/ml-suite/lib/

ln -s libopencv_highgui.so libopencv_highgui.so.3.3
ln -s libopencv_imgcodecs.so libopencv_imgcodecs.so.3.3
ln -s libopencv_imgproc.so libopencv_imgproc.so.3.3
ln -s libopencv_core.so libopencv_core.so.3.3

cd -
