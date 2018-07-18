##############################
# C API WRAPPER
import os
moddir = os.path.dirname(__file__)

import ctypes
lib = ctypes.cdll.LoadLibrary('%s/libnms.so' % moddir)

class BBOX(ctypes.Structure):
    _fields_ = [("classid", ctypes.c_int),
                ("prob", ctypes.c_float),
                ("xlo", ctypes.c_int),
                ("xhi", ctypes.c_int),
                ("ylo", ctypes.c_int),
                ("yhi", ctypes.c_int)]


def do_baseline_nms(conv_out, im_w, im_h, net_w, net_h, out_w, out_h, bboxplanes, classes, scorethresh, iouthresh):

    numbb = ctypes.c_int()
    bboxes_p = ctypes.c_void_p()
    
    lib.do_nms.argtypes = [ctypes.c_float*len(conv_out),
                           ctypes.c_int, # num
                           ctypes.c_int, # imw
                           ctypes.c_int, # imh
                           ctypes.c_int, # netw
                           ctypes.c_int, # neth
                           ctypes.c_int, # outw
                           ctypes.c_int, # outh
                           ctypes.c_int, # bboxplanes
                           ctypes.c_int, # classes
                           ctypes.c_float,  # scorethreshold
                           ctypes.c_float,  # iouthreshold
                           ctypes.c_void_p, # numb (out)
                           ctypes.c_void_p] # bboxes (out)

    lib.do_nms((ctypes.c_float * len(conv_out))(*conv_out),
               len(conv_out),
               im_w,
               im_h,
               net_w,
               net_h,
               out_w,
               out_h,
               bboxplanes,
               classes,
               scorethresh,
               iouthresh,
               ctypes.byref(numbb),
               ctypes.byref(bboxes_p))

    bboxes = ctypes.cast(bboxes_p, ctypes.POINTER(BBOX * numbb.value))

    bboxlist = []
    for x in range(0, numbb.value):
        bboxlist.append({'classid' : bboxes.contents[x].classid,
                         'prob' : bboxes.contents[x].prob,
                        'll' : {'x' : bboxes.contents[x].xlo,
                                'y' : bboxes.contents[x].ylo},
                         'ur' : {'x' : bboxes.contents[x].xhi,
                                 'y' : bboxes.contents[x].yhi}})

    lib.free_bboxes(bboxes_p)
        
    return bboxlist
    
##############################
