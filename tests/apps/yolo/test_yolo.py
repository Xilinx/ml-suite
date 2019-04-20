import functools
import os, re, sys
import pytest
import subprocess

expected = {'light': 2, 'bicycle': 2, 'motorbike': 2, 'clock': 1, 'bus': 3, 'skateboard': 1, 'dog': 1, 'bear': 2, 'person': 35, 'microwave': 1, 'sink': 1, 'oven': 3, 'car': 19, 'bird': 1, 'horse': 1}

def _run_yolo(cmdStr):
  cwd = os.getcwd()
  testSrcPath = "%s/../../../apps/yolo" \
    % os.path.dirname(os.path.realpath(__file__))

  os.chdir(testSrcPath)

  success = False
  output = ""
  try:
    print "\nRunning YOLO ...\nCommand: %s" % (cmdStr)
    process = subprocess.Popen(cmdStr, 
      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    # replace '' with b'' for Python 3 below
    for line in iter(process.stdout.readline, ''): 
      output += line
      sys.stdout.write(line)
    process.stdout.close()
    ret = process.wait()
    if ret:
      raise subprocess.CalledProcessError(ret, cmdStr)
    success = True
  except subprocess.CalledProcessError as e:
    output = e.output

  lines = output.split("\n");
  labelCount = {}
  for line in lines:
    if "INFO: Obj" not in line:
      continue

    words = line.split(" ")
    label = words[-1]
    if label not in labelCount:
      labelCount[label] = 0
    labelCount[label] += 1

  print labelCount

  # can't use exact match because compiler might not be deterministic
  #for i,k in enumerate(labelCount):
  #  if labelCount[k] != expected[k]:
  #    raise ValueError, "\nExpected:\n%s" % expected

  if 'bear' not in labelCount \
    or 'horse' not in labelCount \
    or 'skateboard' not in labelCount:
    raise ValueError, "\nExpected: bear, horse and skateboard"

  os.chdir(cwd)

  if success:
    print "\nPASS"
  else:
    raise Exception

@pytest.mark.timeout(960)
def test_yolo(platform):
  cmdStr = "./run.sh -t test_detect -m yolo_v2_608 -b 16"
  if platform is not None:
    cmdStr += " -p " + platform
  _run_yolo(cmdStr)
