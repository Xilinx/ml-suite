/*
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
# (C) Copyright 2017, Joseph Redmon
# (C) Copyright 2018, Xilinx, Inc.
#
*/
#include "box_nms.h"
#include <stdlib.h>

typedef struct{
    int index;
    int class;
    float **probs;
} box_item_t;

static float box_overlap_1d(float x1, float w1, float x2, float w2);
static float box_intersection(nms_box_t a, nms_box_t b);
static float box_union(nms_box_t a, nms_box_t b);
static float box_iou(nms_box_t a, nms_box_t b);
static int   box_comparator(const void *pa, const void *pb);

/*
 * Find length overlap along one dimension of box.
 * Negative length means no overlap.
 */
static float box_overlap_1d(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

/*
 * Find box intersection
 */
static float box_intersection(nms_box_t a, nms_box_t b)
{
    float w = box_overlap_1d(a.x, a.w, b.x, b.w);
    float h = box_overlap_1d(a.y, a.h, b.y, b.h);
    if(w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
}

/*
 * Find box union
 */
static float box_union(nms_box_t a, nms_box_t b)
{
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
}

/*
 * Find box Intersection over Union (IoU)
 */
static float box_iou(nms_box_t a, nms_box_t b)
{
    return box_intersection(a, b)/box_union(a, b);
}

/*
 * Compare box probablities.
 * Used as callback function for qsort
 * to sort in descending order.
 *
 * a > b, return -1
 * a = b, return  0
 * a < b, return  1
 */
static int box_comparator(const void *pa, const void *pb)
{
    box_item_t a = *(box_item_t *)pa;
    box_item_t b = *(box_item_t *)pb;
    float diff = a.probs[a.index][b.class] - b.probs[b.index][b.class];
    if(diff < 0) return 1;
    else if(diff > 0) return -1;
    return 0;
}

/*
 * Perform intra-frame NMS the simple way.
 */
void box_intra_nms(nms_box_t *boxes, float **probs, int numBoxes, int numClasses, float iouThresh)
{
    int i, j, k;
    for(i = 0; i < numBoxes; ++i){
        int any = 0;
        for(k = 0; k < numClasses; ++k) any = any || (probs[i][k] > 0);
        if(!any) continue;
        for(j = i+1; j < numBoxes; ++j){
            if (box_iou(boxes[i], boxes[j]) > iouThresh){
                for(k = 0; k < numClasses; ++k){
                    if (probs[i][k] < probs[j][k]) probs[i][k] = 0;
                    else probs[j][k] = 0;
                }
            }
        }
    }
}

/*
 * Perform intra-frame NMS using sort class probability method.
 */
void box_intra_nms_sort(nms_box_t *boxes, float **probs, int numBoxes, int numClasses, float iouThresh)
{
    int i, j, k;
    box_item_t *s = calloc(numBoxes, sizeof(box_item_t));

    for(i = 0; i < numBoxes; ++i){
        s[i].index = i;       
        s[i].class = 0;
        s[i].probs = probs;
    }

    for(k = 0; k < numClasses; ++k){
        for(i = 0; i < numBoxes; ++i){
            s[i].class = k;
        }
        qsort(s, numBoxes, sizeof(box_item_t), box_comparator);
        for(i = 0; i < numBoxes; ++i){
            if(probs[s[i].index][k] == 0) continue;
            nms_box_t a = boxes[s[i].index];
            for(j = i+1; j < numBoxes; ++j){
                nms_box_t b = boxes[s[j].index];
                if (box_iou(a, b) > iouThresh){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }

    free(s);
}

/*
 * Perform intra-frame NMS using sort box probability method.
 */
void box_intra_nms_obj(nms_box_t *boxes, float **probs, int numBoxes, int numClasses, float iouThresh)
{
    int i, j, k;
    box_item_t *s = calloc(numBoxes, sizeof(box_item_t));

    for(i = 0; i < numBoxes; ++i){
        s[i].index = i;       
        s[i].class = numClasses;
        s[i].probs = probs;
    }

    qsort(s, numBoxes, sizeof(box_item_t), box_comparator);
    for(i = 0; i < numBoxes; ++i){
        if(probs[s[i].index][numClasses] == 0) continue;
        nms_box_t a = boxes[s[i].index];
        for(j = i+1; j < numBoxes; ++j){
            nms_box_t b = boxes[s[j].index];
            if (box_iou(a, b) > iouThresh){
                for(k = 0; k < numClasses+1; ++k){
                    probs[s[j].index][k] = 0;
                }
            }
        }
    }
    free(s);
}


