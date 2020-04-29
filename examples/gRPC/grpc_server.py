from __future__ import print_function

import inference_server_pb2
import inference_server_pb2_grpc

import numpy as np


class InferenceServicer(inference_server_pb2_grpc.InferenceServicer):
    def Inference(self, request_iterator, context):
        for request in request_iterator:
            print("Request", self.protoToDict(request))
            yield self.dictToProto({"result": np.zeros((224, 3, 3))})

    def protoToDict(self, listOfArrays):
        result = {}
        for arr in listOfArrays.arrayList:
            name = arr.name
            data = np.frombuffer(arr.raw_data, dtype=np.float32)
            result[name] = data
        return result

    def dictToProto(self, nodes):
        result = inference_server_pb2.ListOfArrays()
        for name, data in nodes:
            arr = result.arrayList.new()
            arr.name = name
            arr.raw_data = data.tobytes()
