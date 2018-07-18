import os,sys

## THIS FILE CANNOT BE USED UNLESS THE USER CLONES the COCOAPI parallel to the MLSUITE
## AND THEY DONWLOAD THE VAL2014 annotations, and images
## Also, the PythonAPI directory needs to be built

# Bring in the COCO API for managing the coco dataset
sys.path.insert(0,os.path.abspath("../../../cocoapi/PythonAPI"))
from pycocotools.coco     import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)


#initialize COCO ground truth api
dataDir='../../../cocoapi'
dataType='val2014'
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api
#resFile='%s/results/%s_%s_fake%s100_results.json'
#resFile = resFile%(dataDir, prefix, dataType, annType)

resFile = './results.json'
cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

