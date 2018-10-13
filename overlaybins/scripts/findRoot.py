#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import os, sys

if len(sys.argv) <= 1:
  path = os.path.abspath(os.getcwd())
else:
  path = os.path.abspath(sys.argv[1])

origPath = path

words2LookFor = ["MLsuite", "suite", "ML", "ml"]
for word2LookFor in words2LookFor:
  path = origPath
  while True:
    if path == "/":
      break
    words = path.split("/")
    if len(words) <= 1:
      break

    leaf = words[-1]
    if word2LookFor in leaf:
      # found root
      print(path)
      sys.exit(0)

    path = os.path.dirname(path)

# not found
print("")
