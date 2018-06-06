#ifndef BOX_EXTRACT_H
#define BOX_EXTRACT_H

#include "nms.h"

void yolov2_box_extract(nms_net_t *networkPtr, float *dataIn, float scoreThresh, nms_box_t *boxesOut, float **scoresOut); 

#endif
