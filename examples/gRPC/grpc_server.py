from __future__ import print_function

import inference_server_pb2
import inference_server_pb2_grpc

import request_wrapper

import numpy as np


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    def __init__(self, fpgaRT, output_buffers, n_streams):
        self.fpgaRT = fpgaRT
        self.output_buffers = output_buffers
        self.n_streams = n_streams
        self.in_index = 0
        self.out_index = 0

    def push(self, request):
        # Convert input format
        request = request_wrapper.protoToDict(request)

        # Send to FPGA
        in_slot = self.in_index % self.n_streams
        print("exec_async", request, self.output_buffers[in_slot])
        self.fpgaRT.exec_async(request,
                               self.output_buffers[in_slot],
                               in_slot)
        print("Done exec_async")
        self.in_index += 1

    def pop(self):
        # Wait for finish signal
        out_slot = self.out_index % self.n_streams
        print("Getting result")
        self.fpgaRT.get_result(out_slot)
        print("Got result")

        # Read output
        response = self.output_buffers[out_slot]
        response = request_wrapper.dictToProto(response)
        self.out_index += 1
        return response

    def Inference(self, request_iterator, context):
        for request in request_iterator:
            # Feed to FPGA
            self.push(request)

            # Get output
            if self.in_index - self.out_index >= self.n_streams:
                yield self.pop()
        while self.in_index - self.out_index > 0:
            yield self.pop()
