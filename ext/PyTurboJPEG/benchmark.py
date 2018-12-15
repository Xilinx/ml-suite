#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import cv2
from turbojpeg import TurboJPEG
import timeit
import os
import multiprocessing as mp
import argparse
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
  try:
    bgr_array = jpeg.decode(in_file.read())
    bgr_array = cv2.resize(bgr_array, (224,224))
    in_file.close()
  except Exception, e:
    bgr_array = None 

parser = argparse.ArgumentParser()
parser.add_argument("--numproc", type=int, default=8)
parser.add_argument("--numiter", type=int, default=1)
parser.add_argument("--dir", type = str )
parser.add_argument("--lib", type = str, default="./lib")
args = parser.parse_args()
# using default library installation
jpeg = TurboJPEG(args.lib + "/libturbojpeg.so")

file_dir = absoluteFilePaths(args.dir)
flist = []
for i in range(args.numiter):
  flist += file_dir

p = mp.Pool(processes=args.numproc)
elapsed = timeit.default_timer()
for i in flist:
   p.apply_async(img_decode, args=(i,))

p.close()
p.join()    
print ("%g img/s\n" % (len(flist) / (timeit.default_timer() - elapsed))  )
