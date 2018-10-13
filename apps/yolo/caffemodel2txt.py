#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#

import sys

from caffe.proto import caffe_pb2

infile = sys.argv[1]

print "Processing Caffemodel %s" % infile

model = caffe_pb2.NetParameter()

model.ParseFromString(open(infile,"rb").read())

print model


