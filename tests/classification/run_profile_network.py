import functools
import json, os, re
from base import TestConfig, OutputVerifier, run_test

###########################################
# auto-generate test functions for pytest
###########################################

def extractHwCounter(output):
  lines = output.split("\n")
  for l in lines:
    match = re.search(r'^.+\s+hw_counter\s+:\s+(.+) ms', l)
    if not match:
      continue
    return match.group(1)

  return ""

def extractGeneralCounter(output):
  lastVal = 0
  lines = output.split("\n")
  for l in lines:
    match = re.search(r'^V3 Hardware general counter: .+\s+(.+) ms', l)
    if not match:
      continue
    val = float(match.group(1))
    lastVal = val - lastVal

  return lastVal

def test_profile_network(platform):
  configs = {
    "googlenet_latency": ("../../examples/deployment_modes/data/googlenet_v1_8b_latency.json",
                          "./data/googlenet_v1_data",
                          "./data/googlenet_v1_8b_xdnnv3.json"),
    "googlenet_norepl": ("../../xfdnn/tools/compile/examples/googlev1.1.96.9.noreplication.json",
                          "./data/googlenet_v1_data",
                          "./data/googlenet_v1_8b_xdnnv3.json"),
    "googlenet_repl": ("../../xfdnn/tools/compile/examples/googlev1.1.96.9.replication.json",
                       "./data/googlenet_v1_tensorflow_data",
                       "./data/googlenet_v1_8b_tensorflow_xdnnv3.json"),
    "resnet_latency": ("../../examples/deployment_modes/data/resnet50_8b_latency.json",
                          "./data/resnet50_tensorflow_data",
                          "./data/resnet50_8b_tensorflow_xdnnv3.json"),
    "resnet_norepl": ("../../xfdnn/tools/compile/examples/resnet50.1.96.9.noreplication.json",
                         "./data/resnet50_tensorflow_data",
                         "./data/resnet50_8b_tensorflow_xdnnv3.json"),
    "resnet_repl": ("../../xfdnn/tools/compile/examples/resnet50.1.96.9.replication.json",
                       "./data/resnet50_tensorflow_data",
                       "./data/resnet50_8b_tensorflow_xdnnv3.json")
  }
  (compilerJsonFile, customDataDir, customQuantFile) = configs["resnet_latency"]
  customNetFilePath = "/tmp/isca_custom_%s.json" % os.path.basename(compilerJsonFile)

  with open(compilerJsonFile) as f:
    compilerJson = json.load(f)

  singleLayerMode = False

  ignored = []
  if singleLayerMode:
    ignored = ["XNMaxPool", "XNAvgPool"]

  cumulativeLayers = []
  for layer in compilerJson['network']:
    cumulativeLayers.append(layer)

    if 'xdnn_kv' not in layer \
      or 'XNOp' not in layer['xdnn_kv']:
      continue

    if layer['xdnn_kv']['XNOp'] in ignored:
      continue

    print "\n********************** Running %s...\n" % layer['name']

    if "dst_full_sect_num" in layer['xdnn_kv']:
      # turn off destination replication (or it will crash)
      layer['xdnn_kv']['dst_full_sect_num'] = 1
      layer['xdnn_kv']['dst_repl_sect_num'] = 0
      layer['xdnn_kv']['dst_repl_unit_num'] = 0
      layer['xdnn_kv']['dst_repl_unit_width'] = 0

    layerJson = {}  
    if singleLayerMode:
      layerJson['network'] = [layer]
    else:
      layerJson['network'] = cumulativeLayers
    
    with open(customNetFilePath, "w") as f:  
      json.dump(layerJson, f, sort_keys=True, indent=4)

    cmdArr = ["./run.sh", 
      "-t", "test_classify", 
      "-k", "v3",
      "-b", "8",
      "-cn", customNetFilePath,
      "-cw", customDataDir,
      "-cq", customQuantFile,
      "-m", "custom",
      "-s", "1",
      "-v"]
    cmdStr = " ".join(cmdArr)

    counterVals = {}
    envsToMeasure = ["DL", "UL", "FL", "MISC", "SYSARR"]
    for eKey in envsToMeasure:
      output = ""
      os.environ["XDNN_READ_HARDWARE_GENERAL_COUNTER"] = "1"
      env = "XDNN_HARDWARE_GENERAL_COUNTER_TIME_" +  eKey
      os.environ[env] = "1"
      try:
        output = run_test("test_isca", cmdStr, None, platform)
      except Exception as e:
        output = str(e)
      del os.environ[env]

      counterVals["HW"] = extractHwCounter(output)
      counterVals[eKey] = extractGeneralCounter(output)

    print "HW_COUNTERS %s: %s %s %s %s %s %s" \
      % (layer['name'], 
        counterVals["HW"],
        counterVals["DL"],
        counterVals["FL"],
        counterVals["UL"],
        counterVals["MISC"],
        counterVals["SYSARR"])

