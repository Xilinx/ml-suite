import functools
import os, re, sys
import pytest
import subprocess

def _run_xdlf(cmdStr, verify):
  cwd = os.getcwd()
  testSrcPath = "%s/../../examples/image_classify" \
    % os.path.dirname(os.path.realpath(__file__))

  os.chdir(testSrcPath)

  success = False
  output = ""
  try:
    print "\nRunning XDLF ...\nCommand: %s" % (cmdStr)
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
    if verify:
      verify(output)

    print "\nPASS"
  else:
    raise Exception

def _checkResults(output):
  if "giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca" in output:
    return 

  raise ValueError, "\nWrong prediction; expected 'panda'\n"

@pytest.mark.timeout(300)
def test_xdlf_cpu():
  cmdStr = "./run.sh -o CPU"
  _run_xdlf(cmdStr, _checkResults)

@pytest.mark.timeout(300)
def test_xdlf_hwemu_v2():
  cmdStr = "./run.sh -o HWEmu -k v2"
  _run_xdlf(cmdStr, _checkResults)

@pytest.mark.timeout(300)
def test_xdlf_hwemu_v3():
  cmdStr = "./run.sh -o HWEmu -k v3"
  _run_xdlf(cmdStr, _checkResults)

@pytest.mark.timeout(300)
def test_xdlf_fpga_v2():
  cmdStr = "./run.sh -o FPGA -k v2"
  _run_xdlf(cmdStr, _checkResults)

@pytest.mark.timeout(300)
def test_xdlf_fpga_v3():
  cmdStr = "./run.sh -o FPGA -k v3"
  _run_xdlf(cmdStr, _checkResults)
