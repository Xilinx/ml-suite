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


def overlap(x1, w1,  x2, w2):
    w1by2 = w1/2
    w2by2 = w2/2
    left  = max(x1 - w1by2, x2 - w2by2)
    right = min(x1 + w1by2, x2 + w2by2)
    return right - left


def cal_iou(box, truth) :
    w = overlap(box[0], box[2], truth[0], truth[2])
    h = overlap(box[1], box[3], truth[1], truth[3])
    if (w<0 or h<0):
        return 0
    inter_area = w * h
    union_area = box[2] * box[3] + truth[2] * truth[3] - inter_area
    
    return inter_area * 1.0 / union_area


def sortSecond(val):
    return val[1]

def  apply_nms(boxes, classes, overlap_threshold):
    
    result_boxes=[]
    box_len = (boxes.shape)[0]
    #print "apply_nms :box shape", boxes.shape
    for k in range(classes):  
        
        key_box_class=[]    
        for i in range(box_len):
            key_box_class.append((i, boxes[i,5+k]))
            #boxes[i][4] = k        
        key_box_class.sort(key = sortSecond, reverse = True)
        
        exist_box=np.ones(box_len)        
        for i in range(box_len):
            
            box_id = key_box_class[i][0]
            
            if(exist_box[box_id] == 0):
                continue
                
            if(boxes[box_id,5 + k] < 0.2):
                exist_box[box_id] = 0;
                continue
            
            result_boxes.append([boxes[box_id,0], boxes[box_id,1], boxes[box_id,2], boxes[box_id,3], k, boxes[box_id,5+k]])
            
            for j in range(i+1, box_len):
                
                box_id_compare = key_box_class[j][0]
                if(exist_box[box_id_compare] == 0):
                    continue
                
                overlap = cal_iou(boxes[box_id_compare,:], boxes[box_id,:])
                if(overlap >= overlap_threshold):
                    exist_box[box_id_compare] = 0
    
    return result_boxes
                
def sigmoid_ndarray(data_in):
    data_in = -1*data_in
    data_in = np.exp(data_in) + 1
    data_in = np.reciprocal(data_in)
    
    return data_in
                 
def process_all_yolo_layers(yolo_layers, classes, anchorCnt, nw_in_width, nw_in_height):
    
    biases =[10,13,16,30,33,23, 30,61,62,45,59,119, 116,90,156,198,373,326]
    
    scale_feature=[]
    out_yolo_layers=[]
    for output_id in range(len(yolo_layers)):
        scale_feature.append([output_id,yolo_layers[output_id].shape[3]])
    
    scale_feature.sort(key = sortSecond, reverse = True)
    
    for output_id in range(len(yolo_layers)):
        
        out_id_process = scale_feature[output_id][0]
        #print "process_all_yolo_layers :layer shape", out_id_process, yolo_layers[out_id_process].shape
        width  = yolo_layers[out_id_process].shape[3]
        height = yolo_layers[out_id_process].shape[2]
        w_range = np.arange(float(width))
        h_range = np.arange(float(height)).reshape(height,1)
        
        
        yolo_layers[out_id_process][:,4::(5+classes),:,:] = sigmoid_ndarray(yolo_layers[out_id_process][:,4::(5+classes),:,:])
        
        yolo_layers[out_id_process][:,0::(5+classes),:,:] = sigmoid_ndarray(yolo_layers[out_id_process][:,0::(5+classes),:,:])
        yolo_layers[out_id_process][:,1::(5+classes),:,:] = sigmoid_ndarray(yolo_layers[out_id_process][:,1::(5+classes),:,:])
        yolo_layers[out_id_process][:,0::(5+classes),:,:] = (yolo_layers[out_id_process][:,0::(5+classes),:,:] + w_range)/float(width)
        yolo_layers[out_id_process][:,1::(5+classes),:,:] = (yolo_layers[out_id_process][:,1::(5+classes),:,:] + h_range)/float(height)
        
        yolo_layers[out_id_process][:,2::(5+classes),:,:] = np.exp(yolo_layers[out_id_process][:,2::(5+classes),:,:])
        yolo_layers[out_id_process][:,3::(5+classes),:,:] = np.exp(yolo_layers[out_id_process][:,3::(5+classes),:,:])
        
        
        
        for ankr_cnt in range(anchorCnt):
            channel_number_box_width = ankr_cnt * (5+classes) + 2
            scale_value_width = float(biases[2*ankr_cnt + 2 * anchorCnt * output_id]) /float(nw_in_width)
            channel_number_box_height = ankr_cnt * (5+classes) + 3
            scale_value_height = float(biases[2*ankr_cnt + 2 * anchorCnt * output_id + 1]) /float(nw_in_height)
            
            yolo_layers[out_id_process][:,channel_number_box_width,:,:] = yolo_layers[out_id_process][:,channel_number_box_width,:,:] * scale_value_width
            yolo_layers[out_id_process][:,channel_number_box_height,:,:] = yolo_layers[out_id_process][:,channel_number_box_height,:,:] * scale_value_height
            
            channel_number_classes = ankr_cnt * (5+classes) + 5
            channel_number_obj_score = ankr_cnt * (5+classes) + 4
            
            for class_id in range(classes):                
                cur_channel = channel_number_classes + class_id                
                yolo_layers[out_id_process][:,cur_channel,:,:] = np.multiply(sigmoid_ndarray(yolo_layers[out_id_process][:,cur_channel,:,:]) , yolo_layers[out_id_process][:,channel_number_obj_score,:,:])
                
        yolo_layer_shape = yolo_layers[out_id_process].shape
        out_yolo_layers.append(yolo_layers[out_id_process])
        
    return out_yolo_layers

def darknet_style_xywh(image_width, image_height, llx,lly,urx,ury):
    # Assumes (llx,ury) is upper left corner, and (urx,lly always bottom right
    dw = 1./(image_width)
    dh = 1./(image_height)
    x = (llx + urx)/2.0 - 1
    y = (lly + ury)/2.0 - 1
    w = urx - llx
    h = lly - ury
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return x,y,w,h

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
