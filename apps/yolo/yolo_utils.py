#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
from __future__ import print_function

import colorsys
import os
import random
import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# To run evaluation on the MS COCO validation set, we must convert to
# MS COCO style boxes, which although not well explained is
# The X,Y Coordinate of the upper left corner of the bounding box
# Followed by the box W
# And the box H
# ll = lower left
# ur = upper right
def cornersToxywh(llx,lly,urx,ury):
    # Assumes (0,0) is upper left corner, and urx always greater than llx
    w = urx - llx
    x = llx
    # Assumes (0,0) is upper left corner, and lly always greater than ury
    h = lly - ury
    y = ury
    #print("llx: %d, lly: %d, urx %d, ury %d"%(llx,lly,urx,ury))
    #print("Becomes:")
    #print("x: %d, y: %d, w %d, h %d"%(x,y,w,h))
    return x,y,w,h
    
def softmax(startidx, inputarr, outputarr, n, stride):
    import math

    i = 0
    sumexp = 0.0
    largest = -1*float("inf")

    assert len(inputarr) == len(outputarr), "arrays must be equal"
    for i in range(0, n):
        if inputarr[startidx+i*stride] > largest:
            largest = inputarr[startidx+i*stride]
    for i in range(0, n):
        e = math.exp(inputarr[startidx+i*stride] - largest)
        sumexp += e
        outputarr[startidx+i*stride] = e
    for i in range(0, n):
        outputarr[startidx+i*stride] /= sumexp

        
def sigmoid(x):
    import math
    #print("Sigmoid called on:")
    #print(x)
    #print("")
    if x > 0 :
        return (1 / (1 + math.exp(-x)))
    else:
        return 1 -(1 / (1 + math.exp(x)))

def generate_colors(classes):
    # added color_file.txt so we can customize bad colors
    fname = "color_file.txt"
    colors = []
    if os.path.isfile(fname) == True:
        cf = open(fname, "r")
        for line in cf:
            line = line.rstrip('\n')
            colors.append(eval(line))
        cf.close()
    else:
        hsv_tuples = [(float(x) / classes, 1., 1.) for x in range(classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        random.seed(10101)  # Fixed seed for consistent colors across runs.
        random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
        random.seed(None)  # Reset seed to default.
        cf = open("color_file.txt", "w")
        for i in range(classes):
            print("(%d,%d,%d)"%(colors[i][0],colors[i][1],colors[i][2]), file=cf)
        cf.close

    return colors

def draw_boxes(iimage,bboxes,names,colors,outpath="out",fontpath="font",display=True):
    
    if os.path.isdir('./out') is False:
        os.makedirs('./out')

    image = Image.open(iimage)
    draw = ImageDraw.Draw(image)
    thickness = (image.size[0] + image.size[1]) / 300
    font = ImageFont.truetype(fontpath + '/FiraMono-Medium.otf',15)
    #font = ImageFont.truetype('font/FiraMono-Medium.otf',15)


#    classidset = set()
#    for j in range(len(bboxes)):
#        classidset.add(bboxes[j]['classid'])
#    colorsmap = dict(zip(classidset,range(len(classidset))))
#    colors = generate_colors(len(classidset))

    for j in range(0, len(bboxes)):
        classid = bboxes[j]['classid']
        label   = '{} {:.2f}'.format(names[bboxes[j]['classid']],bboxes[j]['prob'])
        labelsize = draw.textsize(label,font=font)
        for k in range(thickness):
            draw.rectangle([bboxes[j]['ll']['x']+k, bboxes[j]['ll']['y']+k, bboxes[j]['ur']['x']+k, bboxes[j]['ur']['y']+k],outline=colors[classid])
        draw.rectangle([bboxes[j]['ll']['x'], bboxes[j]['ur']['y'], bboxes[j]['ll']['x']+2*thickness+labelsize[0], bboxes[j]['ur']['y']+thickness+labelsize[1]],fill=colors[classid],outline=colors[classid])
        
    for j in range(0, len(bboxes)):
        classid = bboxes[j]['classid']
        label   = '{} {:.2f}'.format(names[bboxes[j]['classid']],bboxes[j]['prob'])
        labelsize = draw.textsize(label)
        draw.text([bboxes[j]['ll']['x']+2*thickness, bboxes[j]['ur']['y']+thickness], label, fill=(0, 0, 0),font=font)
    
    del draw 

    oimage = iimage.split("/")[-1]
    print("oimage =",oimage)
    image.save(os.path.join(outpath,oimage),quality=90)
    print("Saving new image with bounding boxes drawn as %s" % (os.path.join(outpath,oimage)))
    
    # DISPLAY BOXES
    if os.getenv('DISPLAY') is not None and display:
      img = cv2.imread(os.path.join("out",oimage))
      cv2.imshow(oimage,img)
      #import numpy
      #cv2.imshow(oimage, numpy.array(image))
      cv2.waitKey(0)
      cv2.destroyAllWindows()
