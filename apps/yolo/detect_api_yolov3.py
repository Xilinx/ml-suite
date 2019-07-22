import numpy as np

import nms
import time

from  yolo_utils import process_all_yolo_layers,  apply_nms 
from xfdnn.rt import xdnn_io

def set_config(config):
    config['net_w'] = 608
    config['net_h'] = 608
    config['classes'] = 80
    config['anchorCnt'] = 3
    config['batch_sz'] = 1
    config['iouthresh'] = 0.3
    config['labels'] = 'coco.names'
    config['benchmarking'] = 0
    
    if((config['yolo_model'] == 'tiny_yolo_v3')):
         config['classes'] = 3
         config['anchorCnt'] = 3

    with open(config['labels']) as f:
        names = f.readlines()
        config['names'] = [x.strip() for x in names]
    return config

def correct_region_boxes(boxes_array, x_idx, y_idx, w_idx, h_idx, w, h, net_w, net_h):
    
    new_w = 0;
    new_h = 0;
    #print "x_idx, y_idx, w_idx, h_idx, w, h, net_w, net_h", x_idx, y_idx, w_idx, h_idx, w, h, net_w, net_h
    if ((float(net_w) / float(w)) < (float(net_h) / float(h))) :
        new_w = net_w
        new_h = (h * net_w) / w
    else:
        new_w = (w * net_h) / h;
        new_h = net_h
    
    boxes_array[:,x_idx] = (boxes_array[:,x_idx] - (net_w - new_w) / 2.0 / net_w) / (float(new_w) / net_w);
    boxes_array[:,y_idx] = (boxes_array[:,y_idx] - (net_h - new_h) / 2.0 / net_h) / (float(new_h) / net_h);
    boxes_array[:,w_idx] *= float(net_w) / float(new_w);
    boxes_array[:,h_idx] *= float(net_h) / float(new_h);
    
    return boxes_array

# simple HWC->CHW and mean subtraction/scaling
# returns tensor ready for fpga execute
def det_preprocess(image, dest, net_h, net_w):
    #print "in image for preprosessing:", image.shape, image
    dummy_dest, s = xdnn_io.loadYoloImageBlobFromFile(image,  net_h, net_w)    
    dest[...] = dummy_dest
    #print " prep image:", dest.shape, dest
    

# takes dict of two outputs from XDNN, pixel-conv and bb-output
# returns bounding boxes
def det_postprocess(fpgaOutput, config, image_shape):
    print fpgaOutput[0].shape , fpgaOutput[1].shape,  config['classes'], config['anchorCnt'], config['net_w'], config['net_h']    
    out_yolo_layers = process_all_yolo_layers(fpgaOutput, config['classes'], config['anchorCnt'], config['net_w'], config['net_h'])
    anchorCnt = config['anchorCnt']
    classes =  config['classes']
    
    num_proposals_layer=[0]
    total_proposals = 0
    for layr_idx in range (len(out_yolo_layers)):
        yolo_layer_shape = out_yolo_layers[layr_idx].shape
        #print "layr_idx , yolo_layer_shape", layr_idx , yolo_layer_shape
        out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].reshape(yolo_layer_shape[0], anchorCnt, (5+classes), yolo_layer_shape[2]*yolo_layer_shape[3])
        out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].transpose(0,3,1,2)
        out_yolo_layers[layr_idx] = out_yolo_layers[layr_idx].reshape(yolo_layer_shape[0],yolo_layer_shape[2]*yolo_layer_shape[3] * anchorCnt, (5+classes))           
        #print "layr_idx, final in layer sape, outlayer shape", layr_idx, yolo_layer_shape, out_yolo_layers[layr_idx].shape
        total_proposals += yolo_layer_shape[2]*yolo_layer_shape[3] * anchorCnt
        num_proposals_layer.append(total_proposals)
        
    boxes_array = np.empty([config['batch_sz'], total_proposals, (5+classes)])
    
    
    for layr_idx in range (len(out_yolo_layers)):
        proposal_st = num_proposals_layer[layr_idx]
        proposal_ed = num_proposals_layer[layr_idx + 1]
        #print "proposal_st proposal_ed", proposal_st, proposal_ed
        boxes_array[:,proposal_st:proposal_ed,:] = out_yolo_layers[layr_idx][...]
        
    for i in range(config['batch_sz']):
        boxes_array[i,:,:] = correct_region_boxes(boxes_array[i,:,:], 0, 1, 2, 3, float(image_shape[i][1]), float(image_shape[i][0]), float(config['net_w']), float(config['net_h']))
        detected_boxes = apply_nms(boxes_array[i,:,:], classes, config['iouthresh'])
        
        bboxlist=[]
        for det_idx in range(len(detected_boxes)):
            #print  detected_boxes[det_idx][0], detected_boxes[det_idx][1], detected_boxes[det_idx][2], detected_boxes[det_idx][3], config['names'][detected_boxes[det_idx][4]], detected_boxes[det_idx][5]
            bboxlist.append({'classid' : detected_boxes[det_idx][4],
                               'prob' : detected_boxes[det_idx][5],
                               'll' : {'x' : int((detected_boxes[det_idx][0] - 0.5 *detected_boxes[det_idx][2]) * image_shape[i][1]),
                                       'y' : int((detected_boxes[det_idx][1] + 0.5 *detected_boxes[det_idx][3]) * image_shape[i][0])},
                               'ur' : {'x' : int((detected_boxes[det_idx][0] + 0.5 *detected_boxes[det_idx][2]) * image_shape[i][1]),
                                       'y' : int((detected_boxes[det_idx][1] - 0.5 *detected_boxes[det_idx][3]) * image_shape[i][0])}})
    
              

    return bboxlist 

