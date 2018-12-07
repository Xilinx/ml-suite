import functools
from base import TestConfig, OutputVerifier, run_test

###########################################
# define tests
###########################################

testConfig = TestConfig( models=["googlenet_v1", "resnet50"],
		    								 bitwidths=[8, 16],
               	    		 kernel_types=["med", "large", "v3"])


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
expected["resnet50_8_med"] = \
"""
0.8141 "n02123394 Persian cat"
0.1637 "n02112018 Pomeranian"
0.0101 "n02086079 Pekinese, Pekingese, Peke"
0.0065 "n02085782 Japanese spaniel"
0.0010 "n02112350 keeshond"
"""
expected["googlenet_v1_8_v3"] = \
"""
0.7386 "n02112018 Pomeranian"
0.1021 "n02123394 Persian cat"
0.0318 "n02085620 Chihuahua"
0.0261 "n02492035 capuchin, ringtail, Cebus capucinus"
0.0149 "n02123597 Siamese cat, Siamese"
"""
expected["resnet50_8_v3"] = \
"""
0.4619 "n02123394 Persian cat"
0.2166 "n02112018 Pomeranian"
0.0232 "n04399382 teddy, teddy bear"
0.0135 "n02808440 bathtub, bathing tub, bath, tub"
0.0120 "n02111889 Samoyed, Samoyede"
"""
# Not checking 16b, since it doesn't give a good output

expected["googlenet_v1_8_large"] = expected["googlenet_v1_8_med"]
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
        "-t", "test_classify", 
        "-k", kType,
        "-b", str(bitwidth),
        "-m", model]
      cmdStr = " ".join(cmdArr)
      
      configStr = "%s_%s_%s" % (model, bitwidth, kType)
      testName = "test_classify_%s" % configStr
      
      verifier = None
      if configStr in expected:
        verifier = OutputVerifier(expected[configStr]).verify_predictions
      
      # TODO figure out how to add @pytest.mark.timeout(30)
      test = functools.partial(run_test, testName, cmdStr, verifier)
      globals()[testName] = test

