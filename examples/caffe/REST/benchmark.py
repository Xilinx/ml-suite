# import the necessary packages
from __future__ import print_function

import os,sys,argparse

import requests

def request(rest_api_url="http://localhost:9001/predict", image_path="/home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG"):

  # load the input image and construct the payload for the request
  image = open(image_path, "rb").read()
  payload = {"image": image}

  # submit the request

  total_time = 0
  num_requests = 10
  for _ in xrange(num_requests):
    response = requests.post(rest_api_url, files=payload)
    response.raise_for_status()
    total_time += response.elapsed.total_seconds()

  print(' avg latency: {} ms'.format((total_time*1000)/num_requests))

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='pyXFDNN')
  parser.add_argument('--rest_api_url', default="", help='Url to the REST API eg: http://localhost:9000/predict')
  parser.add_argument('--image_path', default="", help='Path to the image eg: /home/mluser/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/ILSVRC2012_val_00000001.JPEG')

  args = vars(parser.parse_args())

  if args["rest_api_url"] and args["image_path"]:
    request(args["rest_api_url"], args["image_path"])
  else:
    print("Missing arguments, provide --rest_api url <>  and --image_path <>")

