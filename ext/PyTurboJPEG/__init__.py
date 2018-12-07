#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import cv2
import os 
from turbojpeg import TurboJPEG

dir_path = os.path.dirname(os.path.realpath(__file__))
lib_jpeg_turbo = TurboJPEG( dir_path + "/lib/libturbojpeg.so")

def imread ( f ):
    try:
        with open(f, 'rb') as in_file:
            img = lib_jpeg_turbo.decode(in_file.read())            
    except Exception as e:
        print (e)
        print ("Unable to decode %s with TurboJPEG, falling back to OpenCV JPEG decode ..." % f)
        img = cv2.imread  (f)    

    return img
    
