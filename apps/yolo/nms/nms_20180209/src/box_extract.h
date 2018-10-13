/*
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
*/
#ifndef BOX_EXTRACT_H
#define BOX_EXTRACT_H

#include "nms.h"

void yolov2_box_extract(nms_net_t *networkPtr, float *dataIn, float scoreThresh, nms_box_t *boxesOut, float **scoresOut); 

#endif
