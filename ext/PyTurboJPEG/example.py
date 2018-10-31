#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import cv2
from turbojpeg import TurboJPEG
import time
import os
import multiprocessing as mp
# specifying library path explicitly
# jpeg = TurboJPEG(r'D:\turbojpeg.dll')
# jpeg = TurboJPEG('/usr/lib64/libturbojpeg.so')
# jpeg = TurboJPEG('/usr/local/lib/libturbojpeg.dylib')
def absoluteFilePaths(directory):
   dirlist = []
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           dirlist.append( os.path.abspath(os.path.join(dirpath, f)))

   return dirlist

# decoding input.jpg to BGR array
def img_decode (f):
    in_file = open(f, 'rb')
    bgr_array = jpeg.decode(in_file.read())
    bgr_array = cv2.resize(bgr_array, (224,224))
    in_file.close()
    return bgr_array

# using default library installation
jpeg = TurboJPEG('../libjpeg-turbo/lib/libturbojpeg.so')

file_dir = absoluteFilePaths("/tmp/ilsvrc12_img_val/")
elapsed = time.time()
numiter = 50
p = mp.Pool(processes=num_proc)
for i in range(numiter):
    result = p.map(img_decode, file_dir)
    
print ("time:" , (time.time() - elapsed) / (len(file_dir) *numiter ) )
