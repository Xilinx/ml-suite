##############################
# C API WRAPPER
import ctypes
lib = ctypes.cdll.LoadLibrary('./libnms.so')
def do_baseline_nms(conv_out, im_w, im_h, net_w, net_h, thresh):
    
    lib.do_nms.argtypes = [ctypes.c_float*len(conv_out), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    lib.do_nms((ctypes.c_float * len(conv_out))(*conv_out), len(conv_out), im_w, im_h, net_w, net_h, thresh)
##############################

# example func1
yolo_out = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
image_width = 1920
image_height = 1080
network_width = 224;
network_height = 225;
detect_threshold = 0.25;

do_baseline_nms(yolo_out, image_width, image_height, network_width, network_height, detect_threshold)

