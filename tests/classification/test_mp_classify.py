import functools
from base import TestConfig, OutputVerifier, run_test

###########################################
# define tests
###########################################
testConfig = TestConfig(models=["googlenet_v1", "resnet50"],
                        bitwidths=[8, 16],
                        kernel_types=["med", "large", "v3"])

###########################################
# define expected results
###########################################
expected = {}

expected["googlenet_v1_8_v3"] = \
"""
Average accuracy (n=256) Top-1: 66.8%, Top-5: 87.5%
"""

expected["resnet50_8_v3"] = \
"""
Average accuracy (n=256) Top-1: 68.0%, Top-5: 89.5%
"""

expected["resnet50_8_med"] = \
"""
Average accuracy (n=256) Top-1: 68.0%, Top-5: 89.5%
"""

expected["googlenet_v1_16_med"] = \
"""
Average accuracy (n=256) Top-1: 68.4%, Top-5: 88.3%
"""

expected["googlenet_v1_16_large"] = expected["googlenet_v1_16_med"]
expected["resnet50_8_large"] = expected["resnet50_8_med"]

###########################################
# auto-generate test functions for pytest
###########################################
for kType in testConfig._kernel_types:
  for model in testConfig._models:
    for bitwidth in testConfig._bitwidths:
      #not supported
      if kType == "v3" and bitwidth == 16:
        continue
      cmdArr = ["./run.sh", 
        "-t", "streaming_classify", 
        "-k", kType,
        "-b", str(bitwidth),
        "-m", model,
        "-g",
        "-d", "../../models/data/ilsvrc12/ilsvrc12_img_val"]
      cmdStr = " ".join(cmdArr)

      configStr = "%s_%s_%s" % (model, bitwidth, kType)
      testName = "test_mp_classify_%s" % configStr

      verifier = None
      if configStr in expected:
        verifier = OutputVerifier(expected[configStr]).verify_accuracy

      # TODO figure out how to add @pytest.mark.timeout(30)
      test = functools.partial(run_test, testName, cmdStr, verifier)
      globals()[testName] = test

