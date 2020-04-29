from __future__ import print_function

import inference_server_pb2

import numpy as np


def protoToDict(listOfArrays):
    result = {}
    for arr in listOfArrays.arrayList:
        name = arr.name
        data = np.frombuffer(arr.raw_data, dtype=np.float32)
        result[name] = data
    return result


def dictToProto(nodes):
    result = inference_server_pb2.ListOfArrays()
    for name, data in nodes.items():
        arr = result.arrayList.add(name=name,
                                   raw_data=data.tobytes())
    return result
