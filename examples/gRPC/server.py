from __future__ import print_function

from xfdnn.rt import xdnn, xdnn_io
import multiprocessing as mp
import ctypes
import numpy as np


def main():
    # Parse arguments
    parser = xdnn_io.default_parser_args()
    parser.add_argument('--deviceID', type=int, default=0,
                        help='FPGA no. -> FPGA ID to run in case multiple FPGAs')
    args = parser.parse_args()
    args = xdnn_io.make_dict_args(args)

    # Create manager
    if not xdnn.createManager():
        raise Exception("Failed to create manager")

    compilerJSONObj = xdnn.CompilerJsonParser(args['netcfg'])

    # Get input and output shape
    input_shapes = list(map(lambda x: (x), compilerJSONObj.getInputs().itervalues()))
    output_shapes = list(map(lambda x: (x), compilerJSONObj.getOutputs().itervalues()))

    for out_idx in range(len(output_shapes)):
        output_shapes[out_idx][0] = args['batch_sz']

    input_node_names = list(map(lambda x: str(x), compilerJSONObj.getInputs().iterkeys()))
    output_node_names = list(map(lambda x: str(x), compilerJSONObj.getOutputs().iterkeys()))

    num_inputs = len(input_shapes)
    num_outputs = len(output_shapes)

    # Create runtime
    ret, handles = xdnn.createHandle(args['xclbin'], "kernelSxdnn_0", [args["deviceID"]])
    if ret != 0:
        raise Exception("Failed to create handle, return value: {error}".format(error=ret))
    fpgaRT = xdnn.XDNNFPGAOp(handles, args)

    print("Input shapes:", input_shapes)
    print("Input nodes:", input_node_names)
    print("Ouput shapes:", output_shapes)
    print("Ouput nodes:", output_node_names)

    input = [mp.Array(ctypes.c_float, np.zeros(shape)) for shape in input_shapes]
    output = [mp.Array(ctypes.c_float, np.prod(shape)) for shape in input_shapes]
    input_dict = {name: shape for name, shape in zip(input_node_names, input)}
    output_dict = {name: shape for name, shape in zip(output_node_names, output)}

    fpgaRT.exec_async(input_dict, output_dict, 0)
    fpgaRT.get_result(0)
    print(output_dict)


if __name__ == '__main__':
    main()