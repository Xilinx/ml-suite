# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 15:41:28 2019

@author: arunkuma
"""
import sys
#sys.path.insert(0, '/wrk/acceleration/users/arun/caffe/python/')
print sys.argv[1]
sys.path.insert(0, sys.argv[1])
import caffe
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import google.protobuf.text_format as tfmt

net_shape = []
net_parameter = caffe.proto.caffe_pb2.NetParameter()
with open(sys.argv[2], "r") as f:
    tfmt.Merge(f.read(), net_parameter)
    net_shape = net_parameter.layer[0].input_param.shape[0].dim
    
print(net_shape[2], net_shape[3] )
n = caffe.NetSpec()
print(type(n))
n.data = L.ImageData(top='label', include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), transform_param=dict(mirror=False, yolo_height=net_shape[2], yolo_width=net_shape[3]), 
                                      image_data_param=dict(source=sys.argv[4],batch_size=1, shuffle=False,root_folder=sys.argv[5]))
with open(sys.argv[3], 'w') as f:
    f.write(str(n.to_proto()))
    print(n.to_proto())

    
net_parameter = caffe.proto.caffe_pb2.NetParameter()
with open(sys.argv[2], "r") as f, open(sys.argv[3], "a") as g:
    tfmt.Merge(f.read(), net_parameter)
    print("before\n", (net_parameter))
    #L = next(L for L in net_parameter.layer if L.name == 'data')
    print(net_parameter.layer[0])
    print(net_parameter.layer[0].input_param.shape[0].dim)
    #L.ImageData(include=dict(phase=caffe_pb2.Phase.Value('TRAIN')), transform_param=dict(mirror=False, yolo_height=608, yolo_width=608), 
    #                                  image_data_param=dict(source="/home/arunkuma/deephi/Image1.txt",batch_size=1, shuffle=False,root_folder="/wrk/acceleration/shareData/COCO_Dataset/val2014_dummy/"))
    
    del net_parameter.layer[0]
    print("after\n", (net_parameter))
    g.write(tfmt.MessageToString(net_parameter))

