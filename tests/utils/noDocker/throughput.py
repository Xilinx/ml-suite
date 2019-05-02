import os, sys
import json
import math
import argparse


def dfl(layer):
    dbLayer = layer
    if dbLayer['memory_description'] != "ddr_to_ddr":
        print ("[XFDNN] Change layer's mem description to ddr_to_ddr ...")
        dbLayer['memory_description'] = "ddr_to_ddr"

    dbLayer['type'] = "Input"
    # Enable HAT (Tiling)
    dbLayer['xdnn_kv']['HAT'] = "1"
    dbLayer['xdnn_kv']['DEST_AM-Buffer_Offset'] = "-1"
    dbLayer['xdnn_kv']['SRCAM-Buffer_0'] = "0"
    dbLayer['xdnn_kv']['SRCAM-Buffer_1'] = "4096"
    dbLayer['xdnn_kv']['tile_height'] = "2"  
    dbLayer['xdnn_kv']['tile_width']  = "32"
    dbLayer['xdnn_kv']['src_full_sect_num']  = "0"
    dbLayer['xdnn_kv']['src_repl_sect_num']  = "1"
    dbLayer['xdnn_kv']['src_repl_unit_num']  = "7"
    dbLayer['xdnn_kv']['src_repl_unit_width']  = "3"
    # Chage to DB
    dbLayer['xdnn_kv']['slice']  = "0"
    # Change I/O addressing to Dynamic
    dbLayer['xdnn_kv']['destAddrReadFromImgQ']  = "1"
    dbLayer['xdnn_kv']['srcAddrReadFromImgQ']  = "1"
    dbLayer['xdnn_kv']['ddr_src'] = "0x0"
    dbLayer['xdnn_kv']['srcAddDDR'] = "1"
    dbLayer['xdnn_kv']['dstAddDDR'] = "1"
    dbLayer['xdnn_kv']['inaddr'] = "0x0"
    dbLayer['xdnn_kv']['outaddr'] = "0x0"
    # Return DFL
    return dbLayer

def mbl(layer):
    mbLayer = layer
    print "[XFDNN] MBL Mem description : ", mbLayer['memory_description']
    if mbLayer['memory_description'] == "am_to_am":
        print ("[XFDNN] MBL Mem description am_to_am, changing it to ddr_to_am ...")
        mbLayer['memory_description'] = "ddr_to_am"
        # Enable HAT (Tiling)
        mbLayer['xdnn_kv']['HAT'] = "1"
        # Set the MBL to read input from DDR (Address from ImgQ)
        mbLayer['xdnn_kv']['srcAddrReadFromImgQ'] = "1"
        mbLayer['xdnn_kv']['destAddrReadFromImgQ'] = "0"
        mbLayer['xdnn_kv']['srcAddDDR'] = "1"
        mbLayer['xdnn_kv']['dstAddDDR'] = "0"
        mbLayer['xdnn_kv']['tile_height'] = "8"
        mbLayer['xdnn_kv']['tile_width'] = "32"
        mbLayer['xdnn_kv']['SRCAM-Buffer_0'] = "0x8000"
        mbLayer['xdnn_kv']['SRCAM-Buffer_1'] = "0x10000"
        mbLayer['xdnn_kv']['DEST_AM-Buffer_Offset'] = "-1"
        mbLayer['xdnn_kv']['inaddr'] = "0x0"
        mbLayer['xdnn_kv']['outaddr'] = "0x0"
    elif mbLayer['memory_description'] == "ddr_to_am":
        print ("[XFDNN] MBL Mem description : ddr_to_am ...")
        mbLayer['xdnn_kv']['HAT'] = "1"
        mbLayer['xdnn_kv']['srcAddrReadFromImgQ'] = "1"
        mbLayer['xdnn_kv']['srcAddDDR'] = "1"
        mbLayer['xdnn_kv']['inaddr'] = "0x0"
    elif mbLayer['memory_description'] == "ddr_to_ddr":
        print ("[XFDNN] MBL Mem description : ddr_to_ddr ...")
        mbLayer['xdnn_kv']['HAT'] = "1"
        mbLayer['xdnn_kv']['srcAddrReadFromImgQ'] = "1"
        mbLayer['xdnn_kv']['srcAddDDR'] = "1"
        mbLayer['xdnn_kv']['inaddr'] = "0x0"
    else: 
        print ("[XFDNN] MBL Mem description : NA, Must be a Gather layer")
        mbLayer['memory_description'] = "ddr_to_am"
        mbLayer['xdnn_kv']['srcAddrReadFromImgQ'] = "1"
        mbLayer['xdnn_kv']['srcAddDDR'] = "1"
    # Return MBL
    return mbLayer

def createGather (prevInstr, nextLayer):
    gather = {}
    gather['name'] = "mb_input"
    gather['active'] = 1
    gather['xdnn_kv'] = {}
    gather['xdnn_kv']['XNOp'] = "XNGather"
    gather['xdnn_kv']['a0']   = "0"
    gather['xdnn_kv']['b1']   = "1"
    gather['xdnn_kv']['c1']   = "1"
    gather['xdnn_kv']['ddr_src'] = "0x0"
    gather['xdnn_kv']['end_row'] = str(int(prevInstr['xdnn_kv']['outsize_h']) - 1) 
    gather['xdnn_kv']['full_sect_num'] = nextLayer['xdnn_kv']['src_full_sect_num']
    gather['xdnn_kv']['repl_sect_num'] = nextLayer['xdnn_kv']['src_repl_sect_num']
    gather['xdnn_kv']['repl_unit_num'] = nextLayer['xdnn_kv']['src_repl_unit_num']
    gather['xdnn_kv']['repl_unit_width'] = nextLayer['xdnn_kv']['src_repl_unit_width']
    if prevInstr['xdnn_kv']['XNOp'] == "XNMaxPoolPipelined":
        gather['xdnn_kv']['inchan'] = prevInstr['xdnn_kv']['pool_inchan']
    else:
        gather['xdnn_kv']['inchan'] = prevInstr['xdnn_kv']['outchan']
    gather['xdnn_kv']['insize_h'] = prevInstr['xdnn_kv']['outsize_h']
    gather['xdnn_kv']['insize_w'] = prevInstr['xdnn_kv']['outsize_w']
    gather['xdnn_kv']['slice'] = "1"
    gather['xdnn_kv']['srcAddrReadFromImgQ'] = "1"
    gather['xdnn_kv']['start_row'] = "0"
    gather['xdnn_kv']['uram_dest'] = nextLayer['xdnn_kv']['inaddr']
    
    # Return gather
    return gather

def main (args):
    print "[XFDNN] Input JSON  : ", args.i
    print "[XFDNN] Output JSON : ", args.o
    with open(args.i) as f:
        data = json.load(f)
    # Check Unsupported layers
    print ("----------------------------------------")
    print (" Unsupported Layers")
    print ("----------------------------------------")
    print json.dumps(data['unsupported'], indent=4)

    print ("\n----------------------------------------")
    print (" Supported Layers")
    print ("----------------------------------------")
    for layer in data['network']:
        print json.dumps(layer['xdnn_kv']['XNOp'] if layer['name'] is None else layer['name'])

    # Create a new list of instructions
    instructions = []
    instructions[:] = data['network'][:]
    
    idx = 0
    # Skip Input instr (will be removed from throughput JSON, has to be removed)
    if instructions[idx]['type'] == "Input" or instructions[idx]['xdnn_kv']['XNOp'] == "XNGather":
        instructions[idx]['active'] = 0
        idx += 1

    firstlayername = None
    # Check the first instr
    if instructions[idx]['xdnn_kv']['XNOp'] == "XNConv" or instructions[idx]['xdnn_kv']['XNOp'] == "XNMaxPoolPipelined":
        dfLayer = dfl(instructions[idx])
        instructions[idx] = dfLayer
        firstlayername = instructions[idx]['xdnn_kv']['name']
        idx += 1

    ndepLayers = 0
    depLayer = None
    # Correct second instr
    if instructions[idx]['xdnn_kv']['XNOp'] == "XNGather":
        print ("[XFDNN] MBL Mem description : Gather layer: ddr_to_am ...")
        instructions[idx]['xdnn_kv']['srcAddrReadFromImgQ'] = "1"
    else:
        for layer in instructions[2:]:
            for bottoms in layer['bottoms']:
                if bottoms == firstlayername:
                    ndepLayers += 1
                    depLayer = layer

        print "[XFDNN] Total dependent Layers: ", ndepLayers
        if ndepLayers > 1:
            print "[XFDNN] Inserting a Gather layer to get the data back to AM"
            getdatabacktoAMlayer = createGather (instructions[1], depLayer)
        else:
            mbLayer = mbl(instructions[idx])
            instructions[idx] = mbLayer 
            idx += 1
            instructions[idx]['xdnn_kv']['inaddr'] = mbLayer['xdnn_kv']['outaddr']
    
    # Add intermediate buffer info
    intermediate = {}

    # Increase OPS in Output layer (XNScatter)
    for layer in instructions:
        if layer['type'] == "Output":
            scatterops = layer['ops']
            layer['ops'] = (64 * math.ceil(instructions[1]['outputshapes'][1] / 8) * \
                            math.ceil(instructions[1]['outputshapes'][2] / (64 / 8)) * \
                            instructions[1]['outputshapes'][3]) * 2

    # Insert intermediate buffer offset info (output of DFL Block)
    intermediate['name'] = "throughput_mode_interbuf"
    intermediate['active'] = 0
    intermediate['type'] = "ThroughputItermediateBuf"
    intermediate['ops'] =  scatterops # Use Scatter Ops here
    intermediate['xdnn_kv'] = {}
    intermediate['xdnn_kv']['XNOp'] = "XNThroughputInterbuf"
    intermediate['xdnn_kv']['name'] = "throughput_mode_interbuf"

    # Insert in network instr list
    instructions.insert(2, intermediate)
    if ndepLayers > 1:
        instructions.insert(3, getdatabacktoAMlayer)

    # Change layer 1 ops and output shapes
    instructions[1]['ops'] = instructions[0]['ops']
    instructions[1]['outputshapes'] = instructions[0]['outputshapes']

    # Delete 1st instruction (Gather, regardless of active status)
    del instructions[0]

    # Put the network instructions back
    del data['network'][:]
    data['network'] = instructions

    # Add intermediate buffer info
    data['throughput_mode_intermediate_buffers'] = []
    data['throughput_mode_intermediate_buffers'].append({})
    data['throughput_mode_intermediate_buffers'][0]['input_address'] = 0 
    data['throughput_mode_intermediate_buffers'][0]['name'] = "throughput_mode_interbuf"

    # Change Inputs
    data['inputs'][0]['input_name'] = data['inputs'][0]['next_layers'][0]
    
    with open(args.o, 'w') as fout:
        #print json.dumps(data, indent=4, sort_keys=True)
        json.dump(data, fout, indent=4, sort_keys=True)

def extant_file(x):
    """
    'Type' for argparse - checks that file exists but does not open.
    """
    if x == "-":
      # skip file check and allow empty string
      return ""

    if not os.path.exists(x):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise argparse.ArgumentTypeError("{0} does not exist".format(x))
    return x

def defparser ():
    parser = argparse.ArgumentParser(description='Throughput Mode JSON Generator')
    parser.add_argument('--i', help='input .json file', type=extant_file, required=True, metavar="FILE")
    parser.add_argument('--o', help='output .json file', default='throughput.json', required=False, metavar="FILE")
    return parser

if __name__ == '__main__':
    parser = defparser ()
    args = parser.parse_args(sys.argv[1:])
    main (args)