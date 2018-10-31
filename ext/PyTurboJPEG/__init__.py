#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from turbojpeg import TurboJPEG
import cv2
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

lib_jpeg_turbo = TurboJPEG( dir_path + "/lib/libturbojpeg.so")

def imread ( f ):
    img = None
    try:
        with open(f, 'rb') as in_file:
            img = lib_jpeg_turbo.decode(in_file.read())            
        #img = cv2.imread  (f )
    except Exception, e:
        print (e)
        print ("Falling back to OpenCV JPEG decode ...")
        img = cv2.imread  (f)    
   

    return img
    
