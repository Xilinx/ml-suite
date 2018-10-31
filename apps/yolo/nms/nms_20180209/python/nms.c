/*
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
*/
#include <stdio.h>

void do_nms(float *arr, int cnt, int im_w, int im_h, int net_w, int net_h, float threshold) {
  int i;
  printf("Inside baseline nms\n");
  for(i = 0; i < cnt && i < 10; ++i) {
    printf("arr[%d] = %f\n", i, arr[i]);
  }
  printf("image (w,h) = (%d,%d)\n", im_w, im_h);
  printf("network (w,h) = (%d,%d)\n", net_w, net_h);
  printf("threshold = %f\n", threshold);
}
