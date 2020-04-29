from __future__ import print_function

import inference_server_pb2
import inference_server_pb2_grpc

import request_wrapper

import numpy as np


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    def Inference(self, request_iterator, context):
        for request in request_iterator:
            print("Request", request_wrapper.protoToDict(request))
            response = request_wrapper.dictToProto({"result": np.zeros((1024,))})
            print("Response", response)
            yield response
