from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

STACK_CHANNELS = True


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    '''
    This implements the inference service
    '''
    def __init__(self, fpgaRT, output_buffers, n_streams, input_shapes):
        '''
        fpgaRT: fpga runtime
        output_buffers: a list of map from node name to numpy array.
           Store the output.
           The length should be equal to n_streams.
        n_streams: number of concurrent async calls
        input_shapes: map from node name to numpy array shape
        '''
        self.fpgaRT = fpgaRT
        self.output_buffers = output_buffers
        self.n_streams = n_streams
        self.input_shapes = input_shapes

        self.in_index = 0  # Index of next output buffer that is free
        self.out_index = 0  # Index of output buffer next output buffer that is doing inference

    def push(self, request):
        # Convert input format
        request = request_wrapper.protoToDict(request, self.input_shapes, stack=STACK_CHANNELS)

        # Send to FPGA
        in_slot = self.in_index % self.n_streams
        self.fpgaRT.exec_async(request,
                               self.output_buffers[in_slot],
                               in_slot)
        self.in_index += 1

    def pop(self):
        # Wait for finish signal
        out_slot = self.out_index % self.n_streams
        self.fpgaRT.get_result(out_slot)

        # Read output
        response = self.output_buffers[out_slot]
        response = request_wrapper.dictToProto(response)
        self.out_index += 1
        return response

    def Inference(self, request_iterator, context):
        for request in request_iterator:
            # Feed to FPGA
            self.push(request)

            # Start to pull output when the queue is full
            if self.in_index - self.out_index >= self.n_streams:
                yield self.pop()

        # pull remaining output
        while self.in_index - self.out_index > 0:
            yield self.pop()
