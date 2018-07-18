"""MIT License from https://github.com/ysh329/darknet-to-caffe-model-convertor/

Copyright (c) 2015 Preferred Infrastructure, Inc.
Copyright (c) 2015 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
import caffe
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *

DEBUG = True
#log_handler = open('darknet2caffe_convert.log', 'w')
#sys.stdout = log_handler

def darknet2caffe(cfgfile, weightfile, protofile, caffemodel):
    net_info = cfg2prototxt(cfgfile)
    save_prototxt(net_info , protofile, region=False)

    #net = caffe.Net(protofile, caffe.TEST)
    net = caffe.Net(protofile, caffe.TRAIN)
    params = net.params

    blocks = parse_cfg(cfgfile)
    fp = open(weightfile, 'rb')
    header = np.fromfile(fp, count=4, dtype=np.int32)
    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()

    layers = []
    layer_id = 1
    start = 0
    for block in blocks:
        if start >= buf.size:
            break
        print(block['type'])
        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            batch_normalize = int(block['batch_normalize'])
            if block.has_key('name'):
                conv_layer_name = block['name']
                bn_layer_name = '%s-bn' % block['name']
                scale_layer_name = '%s-scale' % block['name']
            else:
                conv_layer_name = 'layer%d-conv' % layer_id
                bn_layer_name = 'layer%d-bn' % layer_id
                scale_layer_name = 'layer%d-scale' % layer_id

            if batch_normalize:
                start = load_conv_bn2caffe(buf, start, params[conv_layer_name], params[bn_layer_name], params[scale_layer_name])
            else:
                start = load_conv2caffe(buf, start, params[conv_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'connected':
            if block.has_key('name'):
                fc_layer_name = block['name']
            else:
                fc_layer_name = 'layer%d-fc' % layer_id
            start = load_fc2caffe(buf, start, params[fc_layer_name])
            layer_id = layer_id + 1
        elif block['type'] == 'maxpool':
            layer_id = layer_id + 1
        elif block['type'] == 'avgpool':
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            layer_id = layer_id + 1
        elif block['type'] == 'shortcut':
            layer_id = layer_id + 1
        elif block['type'] == 'softmax':
            layer_id = layer_id + 1
        elif block['type'] == 'cost':
            layer_id = layer_id + 1
        elif block['type'] == 'reorg':
            layer_id = layer_id + 1
        else:
            print("============== unknow ==============")
            print('unknow layer type %s ' % block['type'])
            layer_id = layer_id + 1
    print('save prototxt to %s' % protofile)
    save_prototxt(net_info , protofile, region=False)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)

def load_conv2caffe(buf, start, conv_param):
    weight = conv_param[0].data
    bias = conv_param[1].data
    #conv_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size - narenk
    #conv_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size - narenk

    inter_val = buf[start+bias.size]
    conv_param[1].data[...] = np.reshape(np.append(buf[start+1:start+bias.size], inter_val), bias.shape); start = start + bias.size
    conv_param[0].data[...] = np.reshape(buf[start+1:start+weight.size+1], weight.shape); start = start + weight.size
    return start

def load_fc2caffe(buf, start, fc_param):
    weight = fc_param[0].data
    bias = fc_param[1].data
    fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start

def load_conv_bn2caffe(buf, start, conv_param, bn_param, scale_param):
    conv_weight = conv_param[0].data
    running_mean = bn_param[0].data
    running_var = bn_param[1].data
    scale_weight = scale_param[0].data
    scale_bias = scale_param[1].data

    scale_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
    scale_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size
    bn_param[0].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
    bn_param[1].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
    bn_param[2].data[...] = np.array([1.0])
    conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + conv_weight.size
    return start

def cfg2prototxt(cfgfile):
    blocks = parse_cfg(cfgfile)

    layers = []
    props = OrderedDict() 
    bottom = 'data'
    layer_id = 1
    topnames = dict()
    for bidx in xrange(len(blocks)):
        block = blocks[bidx]
        if block['type'] == 'net':
            props['name'] = 'Darkent2Caffe'
            props['input'] = 'data'
            props['input_dim'] = ['1']
            props['input_dim'].append(block['channels'])
            props['input_dim'].append(block['height'])
            props['input_dim'].append(block['width'])
            continue
        elif block['type'] == 'convolutional':
            conv_layer = OrderedDict()
            if block.has_key('name'):
                conv_layer['name'] = block['name']
                conv_layer['type'] = 'Convolution'
                conv_layer['bottom'] = bottom
                conv_layer['top'] = block['name']
            else:
                conv_layer['name'] = 'layer%d-conv' % layer_id
                conv_layer['type'] = 'Convolution'
                conv_layer['bottom'] = bottom
                conv_layer['top'] = 'layer%d-conv' % layer_id
            convolution_param = OrderedDict()
            convolution_param['num_output'] = block['filters']
            convolution_param['kernel_size'] = block['size']
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(convolution_param['kernel_size'])/2)
            convolution_param['stride'] = block['stride']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            else:
                convolution_param['bias_term'] = 'true'
            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']

            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                if block.has_key('name'):
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'layer%d-bn' % layer_id
                bn_layer['type'] = 'BatchNorm'
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bottom
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

                scale_layer = OrderedDict()
                if block.has_key('name'):
                    scale_layer['name'] = '%s-scale' % block['name']
                else:
                    scale_layer['name'] = 'layer%d-scale' % layer_id
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = bottom
                scale_layer['top'] = bottom
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                scale_layer['scale_param'] = scale_param
                layers.append(scale_layer)

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            max_layer = OrderedDict()
            if block.has_key('name'):
                max_layer['name'] = block['name']
                max_layer['type'] = 'Pooling'
                max_layer['bottom'] = bottom
                max_layer['top'] = block['name']
            else:
                max_layer['name'] = 'layer%d-maxpool' % layer_id
                max_layer['type'] = 'Pooling'
                max_layer['bottom'] = bottom
                max_layer['top'] = 'layer%d-maxpool' % layer_id
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = block['size']
            pooling_param['stride'] = block['stride']
            pooling_param['pool'] = 'MAX'
            if block.has_key('pad') and int(block['pad']) == 1:
                pooling_param['pad'] = str((int(block['size'])-1)/2)
            max_layer['pooling_param'] = pooling_param
            layers.append(max_layer)
            bottom = max_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            avg_layer = OrderedDict()
            if block.has_key('name'):
               avg_layer['name'] = block['name']
               avg_layer['type'] = 'Pooling'
               avg_layer['bottom'] = bottom
               avg_layer['top'] = block['name']
            else:
               avg_layer['name'] = 'layer%d-avgpool' % layer_id
               avg_layer['type'] = 'Pooling'
               avg_layer['bottom'] = bottom
               avg_layer['top'] = 'layer%d-avgpool' % layer_id
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = 7
            pooling_param['stride'] = 1
            pooling_param['pool'] = 'AVE'
            avg_layer['pooling_param'] = pooling_param
            layers.append(avg_layer)
            bottom = avg_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'region':
            region_layer = OrderedDict()
            if block.has_key('name'):
                region_layer['name'] = block['name']
                region_layer['type'] = 'Region'
                region_layer['bottom'] = bottom
                region_layer['top'] = block['name']
            else:
                region_layer['name'] = 'layer%d-region' % layer_id
                region_layer['type'] = 'Region'
                region_layer['bottom'] = bottom
                region_layer['top'] = 'layer%d-region' % layer_id
                region_param = OrderedDict()
                region_param['anchors'] = block['anchors'].strip()
                region_param['classes'] = block['classes']
                region_param['bias_match'] = block['bias_match']
                region_param['coords'] = block['coords']
                region_param['num'] = block['num']
                region_param['softmax'] = block['softmax']
                region_param['jitter'] = block['jitter']
                region_param['rescore'] = block['rescore']

                region_param['object_scale'] = block['object_scale']
                region_param['noobject_scale'] = block['noobject_scale']
                region_param['class_scale'] = block['class_scale']
                region_param['coord_scale'] = block['coord_scale']

                region_param['absolute'] = block['absolute']
                region_param['thresh'] = block['thresh']
                region_param['random'] = block['random']

            region_layer['region_param'] = region_param
            layers.append(region_layer)
            bottom = region_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            print("block:%s" % block)       
            from_layers = block['layers'].split(',')
            if len(from_layers) == 1:
                prev_layer_id = layer_id + int(from_layers[0])
                bottom = topnames[prev_layer_id]
                topnames[layer_id] = bottom
                layer_id = layer_id + 1
            else:
                prev_layer_id1 = layer_id + int(from_layers[0]) 
                prev_layer_id2 = layer_id + int(from_layers[1])
                print("from_layer: %s" % from_layers)
                print("prev_layer_id1: %s" % prev_layer_id1)
                print("prev_layer_id2: %s" % prev_layer_id2)
                print("layer_id: %s" % layer_id)

                bottom1 = topnames[prev_layer_id1]
                bottom2 = topnames[prev_layer_id2]
                concat_layer = OrderedDict()
                if block.has_key('name'):
                    concat_layer['name'] = block['name']
                    concat_layer['type'] = 'Concat'
                    concat_layer['bottom'] = [bottom1, bottom2]
                    concat_layer['top'] = block['name']
                else:
                    concat_layer['name'] = 'layer%d-concat' % layer_id
                    concat_layer['type'] = 'Concat'
                    concat_layer['bottom'] = [bottom1, bottom2]
                    concat_layer['top'] = 'layer%d-concat' % layer_id
                print("concat_layer: %s" % concat_layer)
                layers.append(concat_layer)
                bottom = concat_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id+1
        elif block['type'] == 'shortcut':
            prev_layer_id1 = layer_id + int(block['from'])
            prev_layer_id2 = layer_id - 1
            bottom1 = topnames[prev_layer_id1]
            bottom2= topnames[prev_layer_id2]
            shortcut_layer = OrderedDict()
            if block.has_key('name'):
                shortcut_layer['name'] = block['name']
                shortcut_layer['type'] = 'Eltwise'
                shortcut_layer['bottom'] = [bottom1, bottom2]
                shortcut_layer['top'] = block['name']
            else:
                shortcut_layer['name'] = 'layer%d-shortcut' % layer_id
                shortcut_layer['type'] = 'Eltwise'
                shortcut_layer['bottom'] = [bottom1, bottom2]
                shortcut_layer['top'] = 'layer%d-shortcut' % layer_id
            eltwise_param = OrderedDict()
            eltwise_param['operation'] = 'SUM'
            shortcut_layer['eltwise_param'] = eltwise_param
            layers.append(shortcut_layer)
            bottom = shortcut_layer['top']
 
            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1           
            
        elif block['type'] == 'connected':
            fc_layer = OrderedDict()
            if block.has_key('name'):
                fc_layer['name'] = block['name']
                fc_layer['type'] = 'InnerProduct'
                fc_layer['bottom'] = bottom
                fc_layer['top'] = block['name']
            else:
                fc_layer['name'] = 'layer%d-fc' % layer_id
                fc_layer['type'] = 'InnerProduct'
                fc_layer['bottom'] = bottom
                fc_layer['top'] = 'layer%d-fc' % layer_id
            fc_param = OrderedDict()
            fc_param['num_output'] = int(block['output'])
            fc_layer['inner_product_param'] = fc_param
            layers.append(fc_layer)
            bottom = fc_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'reorg':
            reshape_layer = OrderedDict()
            if block.has_key('name'):
                avg_layer['name'] = block['name']
                reshape_layer['type'] = 'Reshape'
                reshape_layer['bottom'] = bottom
                avg_layer['top'] = block['name']
            else:
                reshape_layer['name'] = 'layer%d-reshape' % layer_id
                reshape_layer['type'] = 'Reshape'
                reshape_layer['bottom'] = bottom
                reshape_layer['top'] = 'layer%d-reshape' % layer_id
            reshape_param = OrderedDict()
            shape = OrderedDict()
            # TODO: auto shape infer
            shape['dim'] = [1, 2048, 9, 9]
            #shape['dim'] = [1, -1, block['stride'], block['stride']]
            reshape_param['shape'] = shape
            reshape_layer['reshape_param'] = reshape_param
            if DEBUG:
                for k in block:
                    print("block[%s]: %s" % (k, block[k]))
            layers.append(reshape_layer) 

            if DEBUG:
                print("============== reorg =========")
                print("reshape['top']: %s" % (reshape_layer['top']))
                print("layer_id: %s" % layer_id)
                print("bottom: %s" % bottom)
            bottom = reshape_layer['top']
            topnames[layer_id] = bottom#reshape_layer['top']
            layer_id = layer_id + 1
        else:
            print('unknow layer type %s ' % block['type'])
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

    net_info = OrderedDict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('try:')
        print('python darknet2caffe.py yolo.xdnn.nobn.cfg yolo.xdnn.nobn.weights yolo.xdnn.nobn.prototxt yolo.xdnn.nobn.caffemodel') 
        print('')
        print('please add name field for each block to avoid generated name')
        exit()

    cfgfile = sys.argv[1]
    #net_info = cfg2prototxt(cfgfile)
    #print_prototxt(net_info)
    #save_prototxt(net_info, 'tmp.prototxt')
    weightfile = sys.argv[2]
    protofile = sys.argv[3]
    caffemodel = sys.argv[4]
    darknet2caffe(cfgfile, weightfile, protofile, caffemodel)
    #format_data_layer(sys.argv[3])
