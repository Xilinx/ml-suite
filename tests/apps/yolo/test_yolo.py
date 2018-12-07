import functools
import os, re, sys
import pytest
import subprocess

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

  os.chdir(cwd)

  if success:
    print "\nPASS"
  else:
    raise Exception

@pytest.mark.timeout(240)
def test_yolo(platform):
  cmdStr = "./run.sh -t e2e"
  if platform is not None:
    cmdStr += " -p " + platform
  _run_yolo(cmdStr)
