#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
import functools
import os, re, sys
import pytest

from base import TestConfig, OutputVerifier, run_test

###########################################
# define tests
###########################################
testConfig = TestConfig(
  models=["multinet"],
  bitwidths=[8],
  kernel_types=["v3"]) #["med", "large"])

###########################################
# define expected results
###########################################
expected = {}
expected["multinet_8_v3"] = \
"""
0.7386 "n02112018 Pomeranian"
0.1021 "n02123394 Persian cat"
0.0318 "n02085620 Chihuahua"
0.0261 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0149 "n02123597 Siamese cat, Siamese"

0.9026 "n02123394 Persian cat"
0.0846 "n02112018 Pomeranian"
0.0053 "n02086079 Pekinese, Pekingese, Peke"
0.0032 "n02085782 Japanese spaniel"
0.0008 "n04399382 teddy, teddy bear"
"""

###########################################
# auto-generate test functions for pytest
###########################################
for bitwidth in testConfig._bitwidths:
  for kType in testConfig._kernel_types:
    for model in testConfig._models:
      cmdArr = ["./run.sh", 
        "-t", "multinet", 
        "-k", kType,
        "-b", str(bitwidth)]
      cmdStr = " ".join(cmdArr)

      configStr = "%s_%s_%s" % (model, bitwidth, kType)
      testName = "test_multinet_%s" % configStr

      verifier = None
      if configStr in expected:
        verifier = OutputVerifier(expected[configStr]).verify_predictions

      # TODO figure out how to add @pytest.mark.timeout(30)
      mytest = functools.partial(run_test, testName, cmdStr, verifier)
      globals()[testName] = mytest

