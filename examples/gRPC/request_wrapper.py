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
            data = bytes()
            batch_size = input_shapes[name][0]
            channel_size = np.prod(input_shapes[name][1:]) * 4
            for i in range(batch_size):
                for channel in range(3):
                    data += arr.raw_data[i*channel_size:(i+1)*channel_size]
            data = np.frombuffer(data, dtype=np.float32).reshape(input_shapes[name])
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
