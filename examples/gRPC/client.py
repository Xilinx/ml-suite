from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

import grpc
import numpy as np


def generate_requests():
    for _ in range(3):
        request = request_wrapper.dictToProto({"input": np.zeros((3, 224, 224))})
        yield request


def start_client():
    with grpc.insecure_channel('ec2-54-245-214-208.us-west-2.compute.amazonaws.com:5000') as channel:
        stub = inference_server_pb2_grpc.InferenceStub(channel)

        responses = stub.Inference(generate_requests())
        for response in responses:
            print(request_wrapper.protoToDict(response))


if __name__ == '__main__':
    request = request_wrapper.dictToProto({"input": np.zeros((3, 224, 224))})
    start_client()
