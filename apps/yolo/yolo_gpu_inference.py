# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:48:05 2019

@author: arunkuma
"""

import sys
#import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import scipy.misc

import os,sys,timeit,json
from multiprocessing import Process, Queue
from yolo_utils import cornersToxywh,generate_colors,draw_boxes, process_all_yolo_layers, apply_nms
# To control print verbosity
import logging as log
import argparse
# Bring in some utility functions from local file
#from yolo_utils import cornersToxywh,sigmoid,softmax,generate_colors,draw_boxes
import numpy as np

# Bring in a C implementation of non-max suppression
sys.path.append('nms')
import nms
#import PyTurboJPEG
import cv2


# Bring in Xilinx XDNN middleware
print(sys.path)
sys.path.insert(0,'../../')
#from xfdnn.rt import xdnn_io

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if x == "-":
      # skip file check and allow empty string
      return ""

    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x


def darknet_style_xywh(image_width, image_height, llx,lly,urx,ury):
    # Assumes (0,0) is upper left corner, and urx always greater than llx
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
    
    
def prep_image(image_file, net_width, net_height, pix_scale, pad_val, img_transpose, ch_swp):
    
    img = cv2.imread(image_file)
    orig_shape = img.shape
    height, width, __ = img.shape
    newdim = max(height, width)
    scalew = float(width)  / newdim
    scaleh = float(height) / newdim
    maxdim = max(net_width, net_height)
    neww   = int(maxdim * scalew)
    newh   = int(maxdim * scaleh)
    img    = cv2.resize(img, (neww, newh))
    
    if img.dtype != np.float32:
        img = img.astype(np.float32, order='C')
    
    img = img * pix_scale
    
    height, width, channels = img.shape
    newdim = max(height, width)
    letter_image = np.zeros((newdim, newdim, channels))
    letter_image[:, :, :] = pad_val
    if newdim == width:
        letter_image[(newdim-height)/2:((newdim-height)/2+height),0:width] = img
    else:
        letter_image[0:height,(newdim-width)/2:((newdim-width)/2+width)] = img
    
    img = letter_image
    
    img = np.transpose(img, (img_transpose[0], img_transpose[1], img_transpose[2]))
    
    ch = 3*[None]
    ch[0] = img[0,:,:]
    ch[1] = img[1,:,:]
    ch[2] = img[2,:,:]
    img   = np.stack((ch[ch_swp[0]],ch[ch_swp[1]],ch[ch_swp[2]]))
    
    return img, orig_shape
        
    
def yolo_gpu_inference(backend_path,
                       class_names_file,
                       image_dir,
                       deploy_model,
                       weights,
                       out_labels,
                       IOU_threshold,
                       scorethresh,
                       dims,
                       mean_value,
                       pxscale,
                       transpose,
                       channel_swap,
                       yolo_model,
                       num_classes,
                       class_names):
    

    #sys.path.insert(0,'/data/users/Repos/XLNX_Internal_Repos/ristretto_chai/ristretto/python')
    #sys.path.insert(0,'/data/users/arun/ML_retrain_testing/caffe/framework/python')
    #sys.path.insert(0,backend_path)
#    sys.path.insert(0, '/wrk/acceleration/users/arun/caffe/python')
    import caffe
    
    #deploy_model = "../../models/caffe/yolov2/fp32/yolov2_224_without_bn_train_quantized_8Bit.prototxt"
    #deploy_model = "../../models/caffe/yolov2/fp32/yolo_deploy_608.prototxt"
    #weights   = "../../models/caffe/yolov2/fp32/yolov2.caffemodel"
    
    net = caffe.Net(deploy_model, weights, caffe.TEST)
    net_parameter = caffe.proto.caffe_pb2.NetParameter()
    caffe.set_mode_cpu()
    last_layer_name = next(reversed(net.layer_dict))
    
    classes = num_classes
    bboxplanes = 5
    net_w = dims[1]
    net_h = dims[2]
    
    import math
    out_w = int(math.ceil(net_w / 32.0))
    out_h = int(math.ceil(net_h / 32.0))
    groups = out_w*out_h
    coords = 4
    groupstride = 1
    batchstride = (groups) * (classes + coords+1)
    beginoffset = (coords+1) * (out_w * out_h)
    #scorethresh = 0.24
    #iouthresh = 0.3
    iouthresh = IOU_threshold

    #colors = generate_colors(classes)

    #imgDir = "../../xfdnn/tools/quantize/calibration_directory"
    #imgDir = "/wrk/acceleration/shareData/COCO_Dataset/val2014"  
    images = sorted([os.path.join(image_dir,name) for name in os.listdir(image_dir)])
    
    

    for i,img in enumerate(images):
        raw_img, s = prep_image(img,  net_w, net_h, pxscale, 0.5, transpose,channel_swap)
        
        net.blobs['data'].data[...] = raw_img
        net.forward()
        fpgaOutput=[]
        if (yolo_model == 'standard_yolo_v3'):
            fpgaOutput.append(net.blobs['layer81-conv'].data[...])
            fpgaOutput.append(net.blobs['layer93-conv'].data[...])
            fpgaOutput.append(net.blobs['layer105-conv'].data[...])
            
            anchorCnt = 3
            
            print "classes fpgaOutput len", classes, len(fpgaOutput)
            out_yolo_layers = process_all_yolo_layers(fpgaOutput, classes, anchorCnt, net_w, net_h)
            
            num_proposals_layer=[0]
            total_proposals = 0
            
            for layr_idx in range (len(out_yolo_layers)):
                yolo_layer_shape = out_yolo_layers[layr_idx].shape
                print "layr_idx , yolo_layer_shape", layr_idx , yolo_layer_shape
                out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].reshape(yolo_layer_shape[0], anchorCnt, (5+classes), yolo_layer_shape[2]*yolo_layer_shape[3])
                out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].transpose(0,3,1,2)
                out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].reshape(yolo_layer_shape[0],yolo_layer_shape[2]*yolo_layer_shape[3] * anchorCnt, (5+classes))           
                print "layr_idx, final in layer sape, outlayer shape", layr_idx, yolo_layer_shape, out_yolo_layers[layr_idx].shape
                total_proposals += yolo_layer_shape[2]*yolo_layer_shape[3] * anchorCnt
                num_proposals_layer.append(total_proposals)
              
         
            boxes_array = np.empty([1, total_proposals, (5+classes)]) 
          
            for layr_idx in range (len(out_yolo_layers)):
                proposal_st = num_proposals_layer[layr_idx]
                proposal_ed = num_proposals_layer[layr_idx + 1]
                print "proposal_st proposal_ed", proposal_st, proposal_ed
                boxes_array[:,proposal_st:proposal_ed,:] = out_yolo_layers[layr_idx][...]
                
            
            boxes_array[0,:,:] = correct_region_boxes(boxes_array[0,:,:], 0, 1, 2, 3, float(s[1]), float(s[0]), float(net_w), float(net_h))
            detected_boxes = apply_nms(boxes_array[i,:,:], classes, iouthresh)
              
            bboxes=[]
            for det_idx in range(len(detected_boxes)):
                print  detected_boxes[det_idx][0], detected_boxes[det_idx][1], detected_boxes[det_idx][2], detected_boxes[det_idx][3], class_names[detected_boxes[det_idx][4]], detected_boxes[det_idx][5]
                
                bboxes.append({'classid' : detected_boxes[det_idx][4],
                                  'prob' : detected_boxes[det_idx][5],
                                  'll' : {'x' : int((detected_boxes[det_idx][0] - 0.5 *detected_boxes[det_idx][2]) * job['shapes'][i][1]),
                                          'y' : int((detected_boxes[det_idx][1] + 0.5 *detected_boxes[det_idx][3]) * job['shapes'][i][0])},
                                  'ur' : {'x' : int((detected_boxes[det_idx][0] + 0.5 *detected_boxes[det_idx][2]) * job['shapes'][i][1]),
                                          'y' : int((detected_boxes[det_idx][1] - 0.5 *detected_boxes[det_idx][3]) * job['shapes'][i][0])}})
                
                       
        else:
            data=net.blobs[last_layer_name].data[...]
            gpu_out= np.copy(data)
            #print("gpu_out.shape = ", gpu_out.shape)
            softmaxout = gpu_out.flatten()
            #print("softmaxout.shape = ", softmaxout.shape)
            
            # first activate first two channels of each bbox subgroup (n)
            for b in range(bboxplanes):
                for r in range(batchstride*b, batchstride*b+2*groups):
                    softmaxout[r] = sigmoid(softmaxout[r])
                
                for r in range(batchstride*b+groups*coords, batchstride*b+groups*coords+groups):
                    softmaxout[r] = sigmoid(softmaxout[r])
                    
            # Now softmax on all classification arrays in image
            for b in range(bboxplanes):
                for g in range(groups):
                    softmax(beginoffset + b*batchstride + g*groupstride, softmaxout, softmaxout, classes, groups)
                    
            
            # NMS
            bboxes = nms.do_baseline_nms(softmaxout,
                                         s[1],
                                         s[0],
                                         net_w,
                                         net_h,
                                         out_w,
                                         out_h,
                                         bboxplanes,
                                         classes,
                                         scorethresh,
                                         iouthresh)
    
        out_line_list = []
    
        filename = img
        out_file_txt = ((filename.split("/")[-1]).split(".")[0])
        #out_file_txt = "/wrk/acceleration/shareData/COCO_Dataset/gpu_val_result_224"+"/"+out_file_txt+".txt"
        out_file_txt = out_labels+"/"+out_file_txt+".txt"
        
        for j in range(len(bboxes)):
            print("Obj %d: %s" % (j, class_names[bboxes[j]['classid']]))
            print("\t score = %f" % (bboxes[j]['prob']))
            print("\t (xlo,ylo) = (%d,%d)" % (bboxes[j]['ll']['x'], bboxes[j]['ll']['y']))
            print("\t (xhi,yhi) = (%d,%d)" % (bboxes[j]['ur']['x'], bboxes[j]['ur']['y']))
        
            x,y,w,h = darknet_style_xywh(s[1], s[0], bboxes[j]["ll"]["x"],bboxes[j]["ll"]["y"],bboxes[j]['ur']['x'],bboxes[j]['ur']['y'])
        
            line_string = str(bboxes[j]["classid"])
            line_string = line_string+" "+str(round(bboxes[j]['prob'],3))
            line_string = line_string+" "+str(x)
            line_string = line_string+" "+str(y)
            line_string = line_string+" "+str(w)
            line_string = line_string+" "+str(h)	
            out_line_list.append(line_string+"\n")
        
        print("loogging into file :", out_file_txt)
        with open(out_file_txt, "w") as the_file:
            for lines in out_line_list:
                the_file.write(lines)
    #draw_boxes(images[i],bboxes,class_names,colors)
    
    return len(images)
    
        
def main():
    parser = argparse.ArgumentParser(description='yolo_gpu_inference')
    parser.add_argument('--class_names_file', help='File with name of the classes in rows', required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--images', help='Direcotry path containing images',
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--deploy_model', help="network definition prototxt file in case of caffe",
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--weights', help="network weights caffe model file in case of caffe",
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--out_labels', help="Direcotry path for stroring output labels in darknet style",
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument('--IOU_threshold', type=float, default=0.3, help='IOU threshold generally set to 0.5')
    parser.add_argument('--prob_threshold', type=float, default=0.005, help='threshold for calculation of f1 score')
    
    parser.add_argument('--dims',type=int, nargs=3,default=[3,224,224] ,help="Dimensions for first layer, default 3,224,224" )
    parser.add_argument('--mean_value', type=int, nargs=3, default=[0,0,0],  # BGR for Caffe
        help='image mean values ')
    parser.add_argument('--pxscale', type=float, default=(1.0/255.0), help='pix cale value')
    parser.add_argument('--transpose', type=int, default=[2,0,1], nargs=3, help="Passed to caffe.io.Transformer function set_transpose, default 2,0,1" )
    parser.add_argument('--channel_swap', type=int, default=[2,1,0], nargs=3, help="Passed to caffe.io.Transformer function set_channel_swap, default 2,1,0")
    parser.add_argument('--backend_path', help='caffe backend',
                        required=True, type=extant_file, metavar="FILE")
    parser.add_argument("--yolo_model",  type=str, default='xilinx_yolo_v2')
    
    

    args = parser.parse_args()
    try:
        args_dict = vars(args)
    except:
        args_dict = args
        
    with open(args_dict['class_names_file']) as f:
        names = f.readlines()
    
    class_names = [x.strip() for x in names]
    print("Class names are  :", class_names)
    num_classes = len(class_names)
    
    num_images_processed = yolo_gpu_inference(args_dict['backend_path'],
                       args_dict['class_names_file'],
                       args_dict['images'],
                       args_dict['deploy_model'],
                       args_dict['weights'],
                       args_dict['out_labels'],
                       args_dict['IOU_threshold'],
                       args_dict['prob_threshold'],
                       args_dict['dims'],
                       args_dict['mean_value'],
                       args_dict['pxscale'],
                       args_dict['transpose'],
                       args_dict['channel_swap'],
                       args_dict['yolo_model'],
                       num_classes,
                       class_names
                       )
    
    print('num images processed : ', num_images_processed)


if __name__ == '__main__':
    main()

        
        
        
        
