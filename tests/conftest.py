# content of conftest.py
import pytest

def pytest_addoption(parser):
  parser.addoption(
      "--platform", action="store", default=None, help="Help guide run.sh with the correct platform info when auto-detection doesn't work"
  )
#   parser.addoption(
#       "--kern_type", action="store", default=["med", "large"], help="my option: type1 or type2"
#   )
#   
#   parser.addoption(
#       "--models", action="store", default=["googlenet_v1", "resnet50"], help="my option: type1 or type2"
#   )
#   
#   parser.addoption(
#       "--bw", action="store", default=[8,16], help="my option: type1 or type2"
#   )    

@pytest.fixture
def platform(request):
    return request.config.getoption("--platform")
  
# @pytest.fixture
# def kern_type(request):
#     return request.config.getoption("--kern_type")  
#   
# @pytest.fixture
# def models(request):
#     return request.config.getoption("--models")
# 
# @pytest.fixture
# def bw(request):
#     return request.config.getoption("--bw")      
