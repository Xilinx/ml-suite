import os, re, sys
import subprocess

class TestConfig():
  __test__ = False

  def __init__(self, models=[], bitwidths=[], kernel_types=[]):
    self._models = models
    self._bitwidths = bitwidths
    self._kernel_types = kernel_types

def run_test(testName, cmdStr, verify, platform):
  if platform is not None:
    cmdStr += " -p " + platform
    
  _testSrcPath = "%s/../../examples/classification" \
    % os.path.dirname(os.path.realpath(__file__))

  cwd = os.getcwd()
  os.chdir(_testSrcPath)

  success = False
  output = ""
  try:
    print "\nRunning [%s] ...\nCommand: %s" % (testName, cmdStr)
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
    if e.output:
      print e.output

  os.chdir(cwd)

  if success:
    if verify:
      verify(output)

    print "\nPASS"
  else:
    raise Exception(output)

  return output

class OutputVerifier():
  def __init__(self, expected):
    self._expected = expected
  
  def get_predictions(self, output):
    predictions = []
    lines = output.split("\n")
    for l in lines:
      match = re.search(r'^([0-9.]+)\s+\"(.+)\"', l)
      if not match:
        continue
      predictions.append([match.group(1), match.group(2)])
  
    return predictions
  
  def verify_predictions(self, output):
    pred = self.get_predictions(output)
    golden = self.get_predictions(self._expected)
  
    if pred != golden:
      raise ValueError, "\nExpected:\n%s" % self._expected

  def get_accuracy(self, output):
    lines = output.split("\n")
    top1 = 0
    top5 = 0
    for l in lines:
      match = re.search(r'^Average accuracy.+Top-1: ([0-9.]+).+Top-5: ([0-9.]+)', l)
      if not match:
        continue
      top1 = match.group(1)
      top5 = match.group(2)

    return (float(top1), float(top5))

  def verify_accuracy(self, output):
    check = self.get_accuracy(output)
    golden = self.get_accuracy(self._expected)

    thresh = 5.
    if golden[0] - check[0] > thresh \
      or golden[1] - check[1] > thresh:
      raise ValueError, "\nExpected:\n%s" % self._expected
