#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "box_extract.h"
#include "box_nms.h"
#include "nms.h"


/*
 * Initialize context and allocate memory
 */
int nms_init(nms_ctx_t *ctxPtr, nms_net_t netCfg, int netMode, nms_cfg_t nmsCfg, int *nmsSeqPtr)
{
    switch (netMode) {
        case NET_CUSTOM:
            break;
        case NET_YOLOV2:
            ctxPtr->boxes = calloc(netCfg.numBoxes, sizeof(nms_box_t));
            ctxPtr->scores = calloc(netCfg.numBoxes, sizeof(float *));
            for (int i=0; i<netCfg.numBoxes; i++) {
                ctxPtr->scores[i] = calloc(netCfg.numClasses+1, sizeof(float *));
            }
            break;
        default:
            return EXIT_FAILURE;
    }

    ctxPtr->network = netCfg;
    ctxPtr->mode    = netMode;
    ctxPtr->nms     = nmsCfg;
    ctxPtr->nmsSeq  = nmsSeqPtr;

    return EXIT_SUCCESS;
}

/*
 * Extract boxes from output layer of network
 */
int nms_extract(nms_ctx_t *ctxPtr, float *dataInPtr) 
{
    switch (ctxPtr->mode) {
        case NET_YOLOV2:
            yolov2_box_extract(&ctxPtr->network, dataInPtr, ctxPtr->nms.mcsScoreThresh, ctxPtr->boxes, ctxPtr->scores);
            break;
        default:
            return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/*
 * Run NMS algorithm
 */
int nms_run(nms_ctx_t *ctxPtr)
{
    int i=0;

    // Execute NMS sequence
    while (ctxPtr->nmsSeq[i] != NMS_UNDEFINED) {
        switch (ctxPtr->nmsSeq[i]) {
            case NMS_MCS:
                box_intra_nms(ctxPtr->boxes, ctxPtr->scores, ctxPtr->network.numBoxes, ctxPtr->network.numClasses, ctxPtr->nms.mcsIoUThresh);
                break;
            case NMS_MCS_OBJ:
                box_intra_nms_obj(ctxPtr->boxes, ctxPtr->scores, ctxPtr->network.numBoxes, ctxPtr->network.numClasses, ctxPtr->nms.mcsIoUThresh);
                break;
            case NMS_MCS_SORT:
                box_intra_nms_sort(ctxPtr->boxes, ctxPtr->scores, ctxPtr->network.numBoxes, ctxPtr->network.numClasses, ctxPtr->nms.mcsIoUThresh);
                break;
            default:
                break;
        }
        i++;
    }

    // TODO: Return number of boxes after prunning, for now it is the same as the original
    return ctxPtr->network.numBoxes;
}

/*
 * Unintialize context and free memory
 */
int nms_uninit(nms_ctx_t *ctxPtr) {
    free(ctxPtr->boxes);
    for (int i=0; i<ctxPtr->network.numBoxes; i++) {
        free(ctxPtr->scores[i]);
    }
    free(ctxPtr->scores);

    return EXIT_SUCCESS;
}
