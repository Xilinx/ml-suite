#ifndef NMS_H
#define NMS_H

typedef enum {
    NET_UNDEFINED,
    NET_CUSTOM,
    NET_YOLOV2,
    NET_INVALID
} nms_net_mode_t;

typedef enum {
    NMS_UNDEFINED,
    NMS_NONE,
    NMS_MCS,
    NMS_MCS_OBJ,
    NMS_MCS_SORT,
    NMS_SEQ,
    NMS_MGP,
    NMS_INVALID
} nms_mode_t;

typedef struct {
    float x, y, w, h;
    float boxScore;
} nms_box_t;

typedef struct {
    float mcsIoUThresh;
    float mcsScoreThresh;
} nms_cfg_t;

typedef struct {
    int imWidth;
    int imHeight;
    int inWidth;
    int inHeight;
    int outWidth;
    int outHeight;
    int numAnchors;
    int numClasses;
    int numBoxes;
    int numCoords;
} nms_net_t;

typedef struct {
    int           mode;
    nms_net_t     network;
    nms_box_t    *boxes;
    float       **scores;
    int          *nmsSeq;
    nms_cfg_t     nms;
} nms_ctx_t;

int nms_init(nms_ctx_t *ctxPtr, nms_net_t netCfg, int netMode, nms_cfg_t nmsCfg, int *nmsSeqPtr);
int nms_extract(nms_ctx_t *ctxPtr, float *dataInPtr);
int nms_run(nms_ctx_t *ctxPtr);
int nms_uninit(nms_ctx_t *ctxPtr);

#endif // NMS_H

