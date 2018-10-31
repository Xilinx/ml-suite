/*
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
*/
#ifndef BOX_NMS_H
#define BOX_NMS_H

#include "nms.h"

void box_intra_nms(nms_box_t *boxes, float **probs, int numBoxes, int numClasses, float iouThresh);
void box_intra_nms_sort(nms_box_t *boxes, float **probs, int numBoxes, int numClasses, float iouThresh);
void box_intra_nms_obj(nms_box_t *boxes, float **probs, int numBoxes, int numClasses, float iouThresh);


#endif

