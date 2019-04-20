#include <boost/algorithm/string.hpp>
typedef struct {
  int classid;
  float prob;
  int xlo;
  int xhi;
  int ylo;
  int yhi;
} bbox;
#ifdef __cplusplus
extern "C"
{ 
#endif
int do_nms(float *arr, int cnt,
           int im_w, int im_h,
           int net_w, int net_h,
           int out_w, int out_h,
           int bboxplanes,
           int classes,
           float scoreThreshold,
           float iouThreshold,
           int *numBoxes, bbox **bboxes) ;
#ifdef __cplusplus 
}
#endif
