#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import os, sys


fpath = os.path.dirname(os.path.realpath(__file__))+"/../"

files = os.listdir(fpath)

abs_files = [fpath + f for f in files]

abs_files = [f for f in abs_files if os.path.isdir(f)]

#print abs_files

for path in abs_files:
  sys.path.insert(0, path)
