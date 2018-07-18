# Parser darknet .cfg file into caffe prototxt and array for input into batch_norm merging
# Issue 1: %s/\'\([^']*\)\'/"\1"/g Changes single to double quotes
# Issue 2: DEBUG - 1x1 Convs should have 0 padding, but darknet cfg has them listed as padding 1. This is because darknet handles this as a special case.


def parse_config(filename):
    # NETWORK PARAMETERS
    layer_num = 0
    inchans = 3
    indims = 224
    is_1x1 = 0
    # MEMORY STORAGE PARAMETERS
    last_layer = 'none'
    layer_dict = {}
    bn = 0 
    f = open(filename)
    for line in f:
        if '#' in line:
            line, comment = line.split('#', 1)
        elif 'channels=' in line:
            inchans = int(line[9:10])
        elif 'width=' in line:
            indims = int(line[6:])    
        elif '[convolutional]' in line:
            last_layer = 'conv'
            layer_dict[layer_num] = {'layer_type': 'conv'} 
            layer_num += 1
        elif '[maxpool]' in line:
            last_layer = 'pool'
            layer_dict[layer_num] = {'layer_type': 'pool'} 
            layer_num += 1
        elif '[route]' in line:
            last_layer = 'route'
            layer_dict[layer_num] = {'layer_type': 'route'} 
            layer_num += 1
        elif '[reorg]' in line:
            sys.exit("You cannot have reorg layers in the config file. Need to be changed to maxpool.")
        elif 'batch_normalize=' in line:
            bn = int(line[16:17])
            layer_dict[layer_num - 1]['bn'] = bn
        elif 'filters=' in line:
            filters = int(line[8:])
            layer_dict[layer_num - 1]['filters'] = filters
            if 'bn' not in layer_dict[layer_num - 1].keys():
                layer_dict[layer_num - 1]['bn'] = 0
        elif 'size=' in line:
            size = int(line[5:])
            if (size==1):
                is_1x1 = 1
            else:
                is_1x1 = 0
            layer_dict[layer_num - 1]['size'] = size
        elif 'stride=' in line:
            stride = int(line[7:])
            layer_dict[layer_num - 1]['stride'] = stride
        elif 'pad=' in line:
            pad = int(line[4:])
            if (is_1x1 == 1):
                pad = 0
            layer_dict[layer_num - 1]['pad'] = pad
        elif 'activation=' in line:
            activation = line[11:]
            layer_dict[layer_num - 1]['activation'] = activation
        elif 'layers=' in line:
            layers = line[7:].split(",")
            layer_dict[layer_num - 1]['layers'] = layers
    layer_dict['net'] = {'channels': inchans, 'dims': indims, 'num_layers': layer_num}          
    #print(layer_dict)
    f.close()
    return layer_dict

def parse_to_array(layer_dict):
    outchan_list = []
    inchan = 3
    conv_layer_out = []
    for l in range(layer_dict['net']['num_layers']):
        if layer_dict[l]['layer_type'] == 'conv':
            #print('{' + str(inchan) + ", " + str(layer_dict[l]['size']) + ", " + str(layer_dict[l]['filters']) + ", " + str(layer_dict[l]['bn']) + '}') 
            conv_layer_out.append([inchan, layer_dict[l]['size'], layer_dict[l]['filters'], layer_dict[l]['bn']])
            inchan = layer_dict[l]['filters']
            outchan_list.append(layer_dict[l]['filters'])  
        if layer_dict[l]['layer_type'] == 'pool':
            outchan_list.append(inchan)    
        if layer_dict[l]['layer_type'] == 'route':
            layers = layer_dict[l]['layers']
            depth_concat = 0
            for i in layers:
                depth_concat += outchan_list[l + int(i)]
            inchan = depth_concat
            outchan_list.append(inchan) 
    return conv_layer_out

def parse_to_prototxt(layer_dict, net_name, file_name):

    layer_list = []
    f = open(file_name, "w")
    
    f.write("name: " + "'" + net_name + "'" + '\n')
    data_layer_1 = "layer " + "{" + '\n' + "  name: " + "'data'" + '\n' + "  type: " + "'Input'" + '\n' + "  top: " + "'data'" + '\n' + "  input_param " + "{ shape: { dim: 1 dim: " + str(layer_dict['net']['channels'])
    data_layer_2 = " dim: " + str(layer_dict['net']['dims']) 
    f.write(data_layer_1 + data_layer_2 + data_layer_2 + " } } " + '\n' + "}" + '\n') 
    
    last_bottom = "data"
    for l in range(layer_dict['net']['num_layers']):
        if layer_dict[l]['layer_type'] == 'conv':
            conv_layer_1 = "layer " + "{" + '\n' + "  name: " + "'conv" + str(l) + "'" + '\n' + "  type: " + "'Convolution'" + '\n' + "  bottom: " + "'" + last_bottom + "'" + '\n' + "  top: " + "'conv" + str(l) + "'" + '\n' 
            conv_layer_2 = "  convolution_param {" + '\n' + "    num_output: " + str(layer_dict[l]['filters']) + '\n' + "    kernel_size: " + str(layer_dict[l]['size']) + '\n' + "    pad: " + str(layer_dict[l]['pad']) + '\n' 
            conv_layer_3 = "    stride: " + str(layer_dict[l]['stride']) + '\n' + "  }" + '\n' + "}" + '\n'
            conv_layer = conv_layer_1 + conv_layer_2 + conv_layer_3
            f.write(conv_layer)

            if layer_dict[l]['activation'] != 'linear\n':
                act_layer_1 = "layer " + "{" + '\n' + "  name: " + "'" + "relu" + str(l) + "'" + '\n' + "  type: " + "'ReLU'" + '\n' + "  bottom: " + "'" + "conv" + str(l) + "'" + '\n'
                act_layer_2 = "  top: " + "'conv" + str(l) + "'" + '\n' + "}" + '\n'
                act_layer = act_layer_1 + act_layer_2
                f.write(act_layer)

            last_bottom = "conv" + str(l)
            layer_list.append(last_bottom)
 
        if layer_dict[l]['layer_type'] == 'pool':
            pool_layer_1 = "layer " + "{" + '\n' + "  name: " + "'pool" + str(l) + "'" + '\n' + "  type: " + "'Pooling'" + '\n' + "  bottom: " + "'" + last_bottom + "'" + '\n' + "  top: " + "'pool" + str(l) + "'" + '\n' 
            pool_layer_2 = "  pooling_param {" + '\n' + "    pool: MAX" + '\n' + "    kernel_size: " + str(layer_dict[l]['size']) + '\n' + "    stride: " + str(layer_dict[l]['stride']) + '\n' 
            pool_layer_3 = "  }" + '\n' + "}" + '\n'
            pool_layer = pool_layer_1 + pool_layer_2 + pool_layer_3
            f.write(pool_layer)

            last_bottom = "pool" + str(l)
            layer_list.append(last_bottom)

        if layer_dict[l]['layer_type'] == 'route':
            layers = layer_dict[l]['layers']
            if len(layers) == 1:
                last_bottom = layer_list[l + int(layers[0])]
                layer_list.append(last_bottom)   
            else:
                concat_layer_1 = "layer " + "{" + '\n' + "  name: " + "'concat" + str(l) + "'" + '\n' + "  type: " + "'Concat'" + '\n'
                concat_layer_mid = ""
                for i in layers:
                    concat_layer_mid += "  bottom: " + "'" + layer_list[l + int(i)] + "'" + '\n'
                concat_layer_3 = "  top: " + "'concat" + str(l) + "'" + '\n' + "}" + '\n'
                concat_layer = concat_layer_1 + concat_layer_mid + concat_layer_3
                f.write(concat_layer)
                last_bottom = "concat" + str(l)
                layer_list.append(last_bottom) 
    f.close()
       
layer_dict = parse_config('yolo-xdnn-tend.cfg')
#arr = parse_to_array(layer_dict)
parse_to_prototxt(layer_dict, "Yolo9000", "yolo_deploy_608.prototxt")
