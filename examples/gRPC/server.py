from __future__ import print_function

from concurrent import futures
import multiprocessing as mp
import ctypes

from xfdnn.rt import xdnn, xdnn_io
import grpc
import numpy as np

import inference_server_pb2_grpc

import grpc_server
import request_wrapper

# Port to listen on
PORT = 5000
# Number of workers for gRPC server
GRPC_WORKER_COUNT = mp.cpu_count()
# Number of concurrent async calls to FPGA
N_STREAMS = 4


# Start a gRPC server
def start_grpc_server(port, fpgaRT, output_buffers, input_shapes):
    print("Starting a gRPC server on port {port}".format(port=port))
    print("Using {n_stream} streams".format(n_stream=N_STREAMS))

    # Configure server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=GRPC_WORKER_COUNT))
    servicer = grpc_server.InferenceServicer(fpgaRT=fpgaRT,
                                             output_buffers=output_buffers,
                                             n_streams=N_STREAMS,
                                             input_shapes=input_shapes)
    inference_server_pb2_grpc.add_InferenceServicer_to_server(servicer,
                                                              server)

    # Bind port
    server.add_insecure_port('[::]:{port}'.format(port=port))

    # Start
    server.start()
    server.wait_for_termination()


def process_inference(request):
    input_dict = request_wrapper.protoToDict(request)


def fpga_init():
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

    for in_idx in range(len(input_shapes)):
        input_shapes[in_idx][0] = args['batch_sz']
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

    print("Batch size:", args['batch_sz'])
    print("Input shapes:", input_shapes)
    print("Input nodes:", input_node_names)
    print("Ouput shapes:", output_shapes)
    print("Ouput nodes:", output_node_names)

    output_buffers = []
    for _ in range(N_STREAMS):
        buffer = {name: np.empty(shape=shape, dtype=np.float32)
                  for name, shape in zip(output_node_names, output_shapes)}
        output_buffers.append(buffer)

    fpgaRT.exec_async({input_node_names[0]: np.zeros(input_shapes[0])},
                      output_buffers[0], 0)
    fpgaRT.get_result(0)

    return fpgaRT, output_buffers, {name: shape for name, shape in zip(input_node_names, input_shapes)}


if __name__ == '__main__':
    fpgaRT, output_buffers, input_shapes = fpga_init()
    start_grpc_server(port=PORT, fpgaRT=fpgaRT, output_buffers=output_buffers, input_shapes=input_shapes)
