#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "box_extract.h"
#include "nms.h"

static int yolov2_entry_index(int w, int h, int classes, int location, int entry);
static nms_box_t yolov2_get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride);
static void yolov2_correct_region_boxes(nms_box_t *boxes, int n, int w, int h, int netw, int neth, int relative);

/*
 * Get data index given location of box
 */
static int yolov2_entry_index(int w, int h, int classes, int location, int entry)
{
    int numCoords = 4;
    int n =   location / (w*h);
    int loc = location % (w*h);
    return n*w*h*(numCoords+classes+1) + entry*w*h + loc;
}


static nms_box_t yolov2_get_region_box(float *x, float *biases, int n, int index, int i, int j, int w, int h, int stride)
{
    nms_box_t b;
    b.x = (i + x[index + 0*stride]) / w;
    b.y = (j + x[index + 1*stride]) / h;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}


static void yolov2_correct_region_boxes(nms_box_t *boxes, int n, int w, int h, int netw, int neth, int relative)
{
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
    } else {
        new_h = neth;
        new_w = (w * neth)/h;
    }
    for (i = 0; i < n; ++i){
        nms_box_t b = boxes[i];
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if(!relative){
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
        }
        boxes[i] = b;
    }
}


void yolov2_box_extract(nms_net_t *networkPtr, float *dataIn, float scoreThresh, nms_box_t *boxesOut, float **scoresOut)
{
    int i,j,n;
    float *predictions = dataIn;
    int numPixels = networkPtr->outWidth * networkPtr->outHeight;

    float biases[] =
    {0.572730004787445068359375f, 
     0.677384972572326660156250f,
     1.874459981918334960937500f,
     2.062530040740966796875000f,
     3.338429927825927734375000f,
     5.474339962005615234375000f,
     7.882820129394531250000000f,
     3.527780055999755859375000f,
     9.770520210266113281250000f,
     9.168279647827148437500000f};

    // Pixel loop
    for (i = 0; i < numPixels; ++i) {
        int row = i / networkPtr->outWidth;
        int col = i % networkPtr->outWidth;

        // Anchor loop
        for(n = 0; n < networkPtr->numAnchors; ++n) {
            int index = n*numPixels + i;

            for(j = 0; j < networkPtr->numClasses; ++j) {
                scoresOut[index][j] = 0;
            }

            int obj_index  = yolov2_entry_index(networkPtr->outWidth, networkPtr->outHeight, networkPtr->numClasses, index, networkPtr->numCoords);
            int box_index  = yolov2_entry_index(networkPtr->outWidth, networkPtr->outHeight, networkPtr->numClasses, index, 0);

            // Get box coordinates
            boxesOut[index] = yolov2_get_region_box(predictions, biases, n, box_index, col, row, networkPtr->outWidth, networkPtr->outHeight, numPixels);
            boxesOut[index].boxScore = predictions[obj_index];

            float max = 0;

            // Class loop, get class probabilities
            for(j = 0; j < networkPtr->numClasses; ++j) {
                int class_index = yolov2_entry_index(networkPtr->outWidth, networkPtr->outHeight, networkPtr->numClasses, index, networkPtr->numCoords + 1 + j);
                float prob = boxesOut[index].boxScore * predictions[class_index];
                scoresOut[index][j] = (prob > scoreThresh) ? prob : 0;
                if(prob > max) max = prob;
            }

            // Set the maximum class score for the last element
            scoresOut[index][networkPtr->numClasses] = max;
        }
    }

    yolov2_correct_region_boxes(boxesOut, networkPtr->numBoxes, networkPtr->imWidth, networkPtr->imHeight, networkPtr->inWidth, networkPtr->inHeight, 1);
}


