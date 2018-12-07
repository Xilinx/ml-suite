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
  kernel_types=["med", "large"])

###########################################
# define expected results
###########################################
expected = {}
expected["multinet_8_med"] = \
"""
0.6735 "n02112018 Pomeranian"
0.1953 "n02123394 Persian cat"
0.0214 "n02123597 Siamese cat, Siamese"
0.0203 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0140 "n02085620 Chihuahua"

0.8141 "n02123394 Persian cat"
0.1637 "n02112018 Pomeranian"
0.0101 "n02086079 Pekinese, Pekingese, Peke"
0.0065 "n02085782 Japanese spaniel"
0.0010 "n02112350 keeshond"
"""
expected["multinet_8_large"] = expected["multinet_8_med"]

###########################################
# auto-generate test functions for pytest
###########################################
for model in testConfig._models:
  for bitwidth in testConfig._bitwidths:
    for kType in testConfig._kernel_types:
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
      test = functools.partial(run_test, testName, cmdStr, verifier)
      globals()[testName] = test

