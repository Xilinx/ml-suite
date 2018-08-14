import numpy as np
import math

def ReLU(x) :
    res = np.copy(x)
    res[np.less(res,0)]=0 
    return res

def Pad_tf(pic, fil_shape, strides, mode, ispool = False, pad_vals = None) :
    pt = None
    pb = None
    pl = None
    pr = None
    print(pic.shape, fil_shape, strides, mode, pad_vals)
    if pad_vals == None :
        ih = pic.shape[1]
        iw = pic.shape[2]
        sh = strides[1]
        sw = strides[2]
        fh = fil_shape[0]
        fw = fil_shape[1]
        if ispool :
            fh = fil_shape[1]
            fw = fil_shape[2]
        if mode == 'NCHW' :
            ih = pic.shape[2]
            iw = pic.shape[3]
            sh = strides[2]
            sw = strides[3]
            fh = fil_shape[2]
            fw = fil_shape[3]
        oh = math.ceil(ih/float(sh))
        ow = math.ceil(iw/float(sw))
        ph = max(0,int((oh-1)*sh+fh-ih))
        pw = max(0,int((ow-1)*sw+fw-iw))
        pt = ph/2
        pb = ph - pt
        pl = pw/2
        pr = pw - pl
        print(oh,ow)
    else :
        pt = pad_vals[0]
        pb = pad_vals[0]
        pl = pad_vals[1]
        pr = pad_vals[1]
    if mode == 'NCHW' :
        return np.pad(pic, ((0,0),(0,0),(pt,pb),(pl,pr)), 'constant')
    else :
        return np.pad(pic, ((0,0),(pt,pb),(pl,pr),(0,0)), 'constant')









