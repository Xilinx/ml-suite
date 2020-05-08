from __future__ import print_function

import inference_server_pb2_grpc

import request_wrapper

import grpc
import numpy as np
import time

# gRPC server info
SERVER_ADDRESS = "localhost"
SERVER_PORT = 5000

# Number of dummy images to send
N_DUMMY_IMAGES = 1000

STACK = True
BATCH_SIZE = 4


def empty_image_generator(n):
    '''
    Generate empty images

    n: number of images
    '''
    for _ in range(n // BATCH_SIZE):
        if STACK:
            request = {"data": np.zeros((BATCH_SIZE, 224, 224), dtype=np.float32)}
        else:
            request = {"data": np.zeros((BATCH_SIZE, 3, 224, 224), dtype=np.float32)}
        request = request_wrapper.dictToProto(request)
        yield request


def dummy_client(n, print_interval=50):
    '''
    Start a dummy client

    n: number of images to send
    print_interval: print a number after this number of images is done
    '''
    print("Dummy client sending {n} images...".format(n=n))

    start_time = time.time()
    # Connect to server
    with grpc.insecure_channel('{address}:{port}'.format(address=SERVER_ADDRESS,
                                                         port=SERVER_PORT)) as channel:
        stub = inference_server_pb2_grpc.InferenceStub(channel)

        # Make a call
        responses = stub.Inference(empty_image_generator(n))

        # Get responses
        for i, response in enumerate(responses):
            if i % print_interval == 0:
                print(i)
            # print(request_wrapper.protoToDict(response,
            #                                   {"loss3_classifier/Reshape_output": (1024)}))
    total_time = time.time() - start_time
    print("{n} images in {time} seconds ({speed} images per second)"
          .format(n=n,
                  time=total_time,
                  speed=float(n) / total_time))


if __name__ == '__main__':
    dummy_client(N_DUMMY_IMAGES)
