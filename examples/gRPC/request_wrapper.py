from __future__ import print_function

import inference_server_pb2

import numpy as np


def protoToDict(listOfArrays, input_shapes, stack=False):
    '''
    Convert a protobuf to a map from node name to data (numpy array)
    '''
    result = {}
    for arr in listOfArrays.arrayList:
        # Node name
        name = arr.name
        # Data
        if stack:
            data = np.frombuffer(arr.raw_data, dtype=np.float32)
            buffer = np.array(input_shapes[name]).reshape((-1))
            size = buffer.size // 3
            buffer[:size] = data
            buffer[size:size*2] = data
            buffer[size*2:] = data
            buffer.reshape(input_shapes[name])
            data = buffer
        else:
            data = np.frombuffer(arr.raw_data, dtype=np.float32).reshape(input_shapes[name])

        result[name] = data
    return result


def dictToProto(nodes):
    '''
    Convert a map from node name to data (numpy array) to protobuf
    '''
    result = inference_server_pb2.ListOfArrays()
    for name, data in nodes.items():
        arr = result.arrayList.add(name=name,
                                   raw_data=data.tobytes())
    return result
