# YOLO Caffe End-to-End Pipeline

# All Related Files (Directory Path: MLsuite/apps/yolo/):
#     caffemodel2txt.py
#     cfg.py 
#     configs.py
#     clean.sh 
#     darknet2caffe.py
#     prototxt.py
#     run.sh
#     xyolo.py
#     yolo_caffe.sh      

# Optional Step 0: Clean entire directory
# ./clean.sh

# Step 1: Port darknet into caffe (.cfg .weights --> .prototxt .caffemodel)
# python darknet2caffe.py yolo-xdnn-tend-bnremove.cfg yolo-xdnn-tend-20180206-bnremove.weights tend.prototxt tend.caffemodel #Debug Help: darknet2caffe_convert.log
# python caffemodel2txt.py tend.caffemodel > tend.txt #Debug Help: Optional 

# Step 2: Comment out region layer for .caffemodel
# Step 3: Change parameters for configurations in configs.py
# Step 4: Select configuration in yolo.py (Line 51) 

# ./run.sh 1525 e2e 
