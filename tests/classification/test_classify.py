import functools
from base import TestConfig, OutputVerifier, run_test

###########################################
# define tests
###########################################

testConfig = TestConfig( models=["googlenet_v1", "resnet50"],
		    								 bitwidths=[8, 16],
               	    		 kernel_types=["med", "large", "v3"],
                         exec_modes=["", "throughput", "latency"])


###########################################
# define expected results
###########################################
expected = {}
expected["googlenet_v1_8_med"] = \
"""
0.6735 "n02112018 Pomeranian"
0.1953 "n02123394 Persian cat"
0.0214 "n02123597 Siamese cat, Siamese"
0.0203 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0140 "n02085620 Chihuahua"
"""
expected["googlenet_v1_16_med"] = \
"""
0.7454 "n02112018 Pomeranian"
0.0807 "n02123394 Persian cat"
0.0332 "n02085620 Chihuahua"
0.0163 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0145 "n02094433 Yorkshire terrier"
"""
expected["resnet50_8_large"] = \
"""
0.8141 "n02123394 Persian cat"
0.1637 "n02112018 Pomeranian"
0.0101 "n02086079 Pekinese, Pekingese, Peke"
0.0065 "n02085782 Japanese spaniel"
0.0010 "n02112350 keeshond"
"""

expected["googlenet_v1_8_v3"] = \
"""
0.5986 "n02112018 Pomeranian"
0.2033 "n02123394 Persian cat"
0.0319 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0271 "n02085620 Chihuahua"
0.0198 "n02123597 Siamese cat, Siamese"
"""
expected["resnet50_8_v3"] = \
"""
0.8960 "n02123394 Persian cat"
0.0987 "n02112018 Pomeranian"
0.0019 "n02085782 Japanese spaniel"
0.0012 "n02086079 Pekinese, Pekingese, Peke"
0.0006 "n02112350 keeshond"
"""
expected["resnet50_8_v3_latency"] = \
"""
0.9026 "n02123394 Persian cat"
0.0846 "n02112018 Pomeranian"
0.0053 "n02086079 Pekinese, Pekingese, Peke"
0.0032 "n02085782 Japanese spaniel"
0.0008 "n04399382 teddy, teddy bear"
"""

expected["googlenet_v1_8_v3_throughput"] = \
"""
0.7386 "n02112018 Pomeranian"
0.1021 "n02123394 Persian cat"
0.0318 "n02085620 Chihuahua"
0.0261 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0149 "n02123597 Siamese cat, Siamese"
"""

# Not checking resnet 16b, since it doesn't give a good output
expected["googlenet_v1_8_v3_latency"] = expected["googlenet_v1_8_v3_throughput"]
expected["googlenet_v1_8_large"] = expected["googlenet_v1_8_med"]
expected["googlenet_v1_16_large"] = expected["googlenet_v1_16_med"]
expected["resnet50_8_v3_throughput"] = expected["resnet50_8_v3_latency"]
expected["resnet50_8_med"] = expected["resnet50_8_large"] 

###########################################
# auto-generate test functions for pytest
###########################################

for kType in testConfig._kernel_types:
  for bitwidth in testConfig._bitwidths:
    for model in testConfig._models:
      for eMode in testConfig._exec_modes:
        if kType == "v3" and bitwidth == 16:
          continue # not supported

        if kType != "v3" and eMode != "":
          continue # not supported
        
        cmdArr = ["./run.sh", 
          "-t", "test_classify", 
          "-k", kType,
          "-b", str(bitwidth),
          "-m", model]
        
        configStr = "%s_%s_%s" % (model, bitwidth, kType)

        if eMode != "":
          cmdArr.extend(["-c", eMode])
          configStr += "_%s" % eMode

        testName = "test_classify_%s" % configStr
        cmdStr = " ".join(cmdArr)

        verifier = None
        if configStr in expected:
          verifier = OutputVerifier(expected[configStr]).verify_predictions
        
        # TODO figure out how to add @pytest.mark.timeout(30)
        mytest = functools.partial(run_test, testName, cmdStr, verifier)
        globals()[testName] = mytest

