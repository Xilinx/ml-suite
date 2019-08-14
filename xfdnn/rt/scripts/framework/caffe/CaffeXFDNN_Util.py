#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import google.protobuf.text_format as pbtf
import caffe.proto.caffe_pb2 as caffe_pb2
import json

from xfdnn.rt.scripts.framework.base.quantize_controls import quantize_controls
#from xfdnn.rt.scripts.framework.caffe.CaffeXFDNN_Quantize import CaffeXFDNN_Quantize

def get_subgraph_nodes(proto) :
    quantized_inputs = set()
    res = []
    for layer in proto.layer:
        if layer.type == "Python" and layer.python_param.module == "xfdnn.rt.scripts.framework.caffe.CaffeXFDNN_UnQuantize":
            continue
        if layer.type == 'Python' and layer.python_param.module == "xfdnn.rt.scripts.framework.caffe.CaffeXFDNN_Quantize":
            for t_name in layer.top :
                quantized_inputs.add(t_name)
        all_inps_quantized = True
        if len(layer.bottom) == 0 :
            all_inps_quantized = False
        for inp in layer.bottom :
            if inp not in quantized_inputs :
                all_inps_quantized = False
                break
        if all_inps_quantized :
            if layer.type in ["Convolution", "Scale"]: 
                res.append(layer.name)
            for t_name in layer.top :
                quantized_inputs.add(t_name)
    return res


def quantize_weights(net, model_def, q_cfg):
    quantize_cfg = None
    with open(q_cfg) as qfile :
        quantize_cfg = json.load(qfile)
    proto = caffe_pb2.NetParameter()
    with open(model_def) as pfile :
        pbtf.Parse(pfile.read(), proto)
    node_names = get_subgraph_nodes(proto)
    env = quantize_controls(q_cfg)
    for name in node_names :
        env.quantize_wts(net.params[name][0].data, name)
        env.quantize_bias(net.params[name][1].data, name)
































