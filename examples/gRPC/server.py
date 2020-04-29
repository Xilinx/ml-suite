from __future__ import print_function

from concurrent import futures
import multiprocessing as mp
import ctypes

from xfdnn.rt import xdnn, xdnn_io
import grpc
import numpy as np

import inference_server_pb2_grpc

import grpc_server

GRPC_WORKER_COUNT = mp.cpu_count()
GRPC_PROCESS_COUNT = mp.cpu_count()
PORT = 5000


# Start a gRPC server
def start_grpc_server(port):
    print("Starting a gRPC server on port {port}".format(port=port))

    # Configure server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=GRPC_WORKER_COUNT))
    inference_server_pb2_grpc.add_InferenceServicer_to_server(grpc_server.InferenceServicer(),
                                                              server)

    # Bind port
    server.add_insecure_port('[::]:{port}'.format(port=port))

    # Start
    server.start()
    server.wait_for_termination()


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

    input = [mp.Array(ctypes.c_float, np.prod(shape)) for shape in input_shapes]
    input_arr = [np.frombuffer(arr.get_obj(), dtype=np.float32).reshape(shape)
                 for arr, shape in zip(input, input_shapes)]
    output = [mp.Array(ctypes.c_float, np.prod(shape)) for shape in output_shapes]
    output_arr = [np.frombuffer(arr.get_obj(), dtype=np.float32).reshape(shape)
                  for arr, shape in zip(output, output_shapes)]
    input_dict = {name: arr for name, arr in zip(input_node_names, input_arr)}
    output_dict = {name: arr for name, arr in zip(output_node_names, output_arr)}

    fpgaRT.exec_async(input_dict, output_dict, 0)
    fpgaRT.get_result(0)
    print(output_dict)


if __name__ == '__main__':
    start_grpc_server(port=PORT)
