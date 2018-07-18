#include <stdio.h>
#include <stdlib.h>
#include "nms.h"

typedef struct {
  int classid;
  float prob;
  int xlo;
  int xhi;
  int ylo;
  int yhi;
} bbox;

void init_bbox(bbox *bbox, int classid, float prob, int xlo, int xhi, int ylo, int yhi) {
  printf("\tCreating bbox (classid,prob) = (%d,%f), (xlo,xhi) = (%d,%d), (ylo,yhi) = (%d,%d)\n", classid, prob, xlo, xhi, ylo, yhi);
  bbox->classid = classid;
  bbox->prob = prob;
  bbox->xlo = xlo;
  bbox->xhi = xhi;
  bbox->ylo = ylo;
  bbox->yhi = yhi;
}


void free_bboxes(float *arr) {
  free(arr);
}


int max_index(float *a, int n) {
  if(n <= 0) return -1;
  int i, max_i = 0;
  float max = a[0];
  for(i = 1; i < n; ++i){
    if(a[i] > max){
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}


int gen_detections(int im_w, int im_h, int planes, int pw, int ph, float thresh, nms_box_t *boxes, float **probs, int classes, bbox *bboxout) {
  int pixel;
  int plane;
  
  int bboxcnt = 0;
  for(plane = 0; plane < planes; ++plane) {
    for(pixel = 0; pixel < pw*ph; ++pixel) {
      int i = plane*(pw*ph) + pixel;
      int class = max_index(probs[i], classes);
      float prob = probs[i][class];
      
      if(prob > thresh){
	
	nms_box_t b = boxes[i];
	
	int left  = (b.x-b.w/2.)*im_w;
	int right = (b.x+b.w/2.)*im_w;
	int top   = (b.y-b.h/2.)*im_h;
	int bot   = (b.y+b.h/2.)*im_h;
	
	if(left < 0) left = 0;
	if(right > im_w-1) right = im_w-1;
	if(top < 0) top = 0;
	if(bot > im_h-1) bot = im_h-1;
	
	
	bboxout[bboxcnt].classid = class;
	bboxout[bboxcnt].prob = prob;
	bboxout[bboxcnt].xlo = left;
	bboxout[bboxcnt].xhi = right;
	bboxout[bboxcnt].ylo = bot;
	bboxout[bboxcnt].yhi = top;
	++bboxcnt;
      }
    }
  }
  return bboxcnt;
}



/* example code for how NMS returns bounding box structures from C/C++ into python */
int do_nms(float *arr, int cnt,
	   int im_w, int im_h,
	   int net_w, int net_h,
	   int out_w, int out_h,
	   int bboxplanes,
	   int classes,
	   float scoreThreshold, 
	   float iouThreshold,
	   int *numBoxes, bbox **bboxes) {

  nms_ctx_t nmsCtx;
  nms_net_t nmsNet;
  nms_cfg_t nmsCfg;
  int       nmsSeq[] = {NMS_MCS_OBJ, NMS_UNDEFINED};
  
  int numcoords = 4;
  
  nmsNet.imWidth         = im_w;
  nmsNet.imHeight        = im_h;
  nmsNet.inWidth         = net_w;
  nmsNet.inHeight        = net_h;
  nmsNet.outWidth        = out_w;
  nmsNet.outHeight       = out_h;
  nmsNet.numAnchors      = bboxplanes;
  nmsNet.numClasses      = classes;
  nmsNet.numBoxes        = out_w*out_h*bboxplanes;
  nmsNet.numCoords       = numcoords;
  
  nmsCfg.mcsIoUThresh    = iouThreshold;
  nmsCfg.mcsScoreThresh  = scoreThreshold;

  nms_init(&nmsCtx, nmsNet, NET_YOLOV2, nmsCfg, nmsSeq);
  nms_extract(&nmsCtx, arr);
  nms_run(&nmsCtx);

  /* Initialize final output max size array */  
  *bboxes = (bbox*)calloc(out_w*out_h*bboxplanes, sizeof(bbox));
  *numBoxes = gen_detections(im_w, im_h, bboxplanes, out_w, out_h, nmsCfg.mcsScoreThresh, nmsCtx.boxes, nmsCtx.scores, classes, *bboxes);

  nms_uninit(&nmsCtx);

  return *numBoxes;
}


