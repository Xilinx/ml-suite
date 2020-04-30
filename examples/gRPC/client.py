from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

import grpc
import numpy as np
import time


def generate_requests():
    try:
        for _ in range(1000):
            request = request_wrapper.dictToProto({"data": np.zeros((3, 224, 224), dtype=np.float32)})
            yield request
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise e


def start_client():
    with grpc.insecure_channel('localhost:5000') as channel:
        stub = inference_server_pb2_grpc.InferenceStub(channel)

        responses = stub.Inference(generate_requests())
        for i, response in enumerate(responses):
            if i % 50 == 0:
                print(i)
            # print(request_wrapper.protoToDict(response,
            #                                   {"loss3_classifier/Reshape_output": (1024)}))


if __name__ == '__main__':
    request = request_wrapper.dictToProto({"data": np.zeros((3, 224, 224))})
    start_time = time.time()
    start_client()
    print(time.time() - start_time)
