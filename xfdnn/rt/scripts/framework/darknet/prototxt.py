#!/usr/bin/env python
#
# (C) Copyright 2018, Xilinx, Inc.
#
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


from collections import OrderedDict
try:
    import caffe.proto.caffe_pb2 as caffe_pb2
except:
    try:
        import caffe_pb2
    except:
        print 'caffe_pb2.py not found. Try:'
        print '  protoc caffe.proto --python_out=.'
        exit()

def parse_caffemodel(caffemodel):
    model = caffe_pb2.NetParameter()
    print 'Loading caffemodel: ', caffemodel
    with open(caffemodel, 'rb') as fp:
        model.ParseFromString(fp.read())

    return model


def parse_prototxt(protofile):
    def line_type(line):
        if line.find(':') >= 0:
            return 0
        elif line.find('{') >= 0:
            return 1
        return -1

    def parse_block(fp):
        block = OrderedDict()
        line = fp.readline().strip()
        while line != '}':
            ltype = line_type(line)
            if ltype == 0: # key: value
                #print line
                line = line.split('#')[0]
                key, value = line.split(':')
                key = key.strip()
                value = value.strip().strip('"')
                if block.has_key(key):
                    if type(block[key]) == list:
                        block[key].append(value)
                    else:
                        block[key] = [block[key], value]
                else:
                    block[key] = value
            elif ltype == 1: # blockname {
                key = line.split('{')[0].strip()
                sub_block = parse_block(fp)
                block[key] = sub_block
            line = fp.readline().strip()
            line = line.split('#')[0]
        return block

    fp = open(protofile, 'r')
    props = OrderedDict()
    layers = []
    line = fp.readline()
    while line != '':
        line = line.strip().split('#')[0]
        if line == '':
            line = fp.readline()
            continue
        ltype = line_type(line)
        if ltype == 0: # key: value
            key, value = line.split(':')
            key = key.strip()
            value = value.strip().strip('"')
            if props.has_key(key):
               if type(props[key]) == list:
                   props[key].append(value)
               else:
                   props[key] = [props[key], value]
            else:
                props[key] = value
        elif ltype == 1: # blockname {
            key = line.split('{')[0].strip()
            if key == 'layer':
                layer = parse_block(fp)
                layers.append(layer)
            else:
                props[key] = parse_block(fp)
        line = fp.readline()

    if len(layers) > 0:
        net_info = OrderedDict()
        net_info['props'] = props
        net_info['layers'] = layers
        return net_info
    else:
        return props

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def print_prototxt(net_info):
    # whether add double quote
    def format_value(value):
        #str = u'%s' % value
        #if str.isnumeric():
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '\"%s\"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' ']*indent)
        print('%s%s {' % (blanks, prefix))
        for key,value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent+4)
            elif type(value) == list:
                for v in value:
                    print('%s    %s: %s' % (blanks, key, format_value(v)))
            else:
                print('%s    %s: %s' % (blanks, key, format_value(value)))
        print('%s}' % blanks)
        
    props = net_info['props']
    layers = net_info['layers']
    print('name: \"%s\"' % props['name'])
    print('input: \"%s\"' % props['input'])
    print('input_dim: %s' % props['input_dim'][0])
    print('input_dim: %s' % props['input_dim'][1])
    print('input_dim: %s' % props['input_dim'][2])
    print('input_dim: %s' % props['input_dim'][3])
    print('')
    for layer in layers:
        print_block(layer, 'layer', 0)

def save_prototxt(net_info, protofile, region=True):
    fp = open(protofile, 'w')
    # whether add double quote
    def format_value(value):
        #str = u'%s' % value
        #if str.isnumeric():
        if is_number(value):
            return value
        elif value == 'true' or value == 'false' or value == 'MAX' or value == 'SUM' or value == 'AVE':
            return value
        else:
            return '\"%s\"' % value

    def print_block(block_info, prefix, indent):
        blanks = ''.join([' ']*indent)
        print >>fp, '%s%s {' % (blanks, prefix)
        for key,value in block_info.items():
            if type(value) == OrderedDict:
                print_block(value, key, indent+4)
            elif type(value) == list:
                for v in value:
                    print >> fp, '%s    %s: %s' % (blanks, key, format_value(v))
            else:
                print >> fp, '%s    %s: %s' % (blanks, key, format_value(value))
        print >> fp, '%s}' % blanks
        
    props = net_info['props']
    layers = net_info['layers']
    
    print >> fp, 'name: \"%s\"' % props['name']
    print >> fp, 'layer {'
    print >> fp, '  name: \"data\"'
    print >> fp, '  type: \"Input\"'
    print >> fp, '  top: \"data\"'
    print >> fp, '  input_param {'
    print >> fp, '    shape {'
    print >> fp, '      dim: %s' % props['input_dim'][0]
    print >> fp, '      dim: %s' % props['input_dim'][1]
    print >> fp, '      dim: %s' % props['input_dim'][2]
    print >> fp, '      dim: %s' % props['input_dim'][3]
    print >> fp, '    }'
    print >> fp, '  }'
    print >> fp, '}'
    """
    print >> fp, 'input: \"%s\"' % props['input']
    print >> fp, 'input_dim: %s' % props['input_dim'][0]
    print >> fp, 'input_dim: %s' % props['input_dim'][1]
    print >> fp, 'input_dim: %s' % props['input_dim'][2]
    print >> fp, 'input_dim: %s' % props['input_dim'][3]
    """
    
    print >> fp, ''
    for layer in layers:
        if layer['type'] != 'Region' or region == True:
            print_block(layer, 'layer', 0)
    fp.close()


def format_data_layer(protofile):
    model_name_pattern = '(.*)\..*'
    dim_pattern = 'input_dim: (.*)'
    with open(protofile) as protofile_handle:
        lines = protofile_handle.readlines()
   
    try:
        import re
        #model_name = re.findall(model_name_pattern, protofile)[0]
        model_name = re.findall(model_name_pattern, protofile.replace("/", "-"))[0]
        dim = [re.findall(dim_pattern, lines[1])[0],
               re.findall(dim_pattern, lines[2])[0],
               re.findall(dim_pattern, lines[3])[0],
               re.findall(dim_pattern, lines[4])[0],
              ]
    except:
        print("Don't need to format data layer")
        savefile_handle.close()
        return

    dim = map(str, dim)
    data_layer_str = '''name: "%(model_name)s"

layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: %(dim0)s
      dim: %(dim1)s
      dim: %(dim2)s
      dim: %(dim3)s
    }    
  }
}\n''' % {'model_name': model_name, 'dim0': dim[0], 'dim1': dim[1], 'dim2': dim[2], 'dim3': dim[3]}

    print(data_layer_str)

    proto_lines_str = data_layer_str + reduce(lambda l1,l2: l1+l2, lines[5:])

    savefile_handle = open(protofile, "w")
    savefile_handle.write(proto_lines_str)    
    savefile_handle.close()
        

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print('Usage: python prototxt.py model.prototxt')
        exit()

    net_info = parse_prototxt(sys.argv[1])
    print_prototxt(net_info)
    save_prototxt(net_info, 'tmp.prototxt')
