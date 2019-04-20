# -*- coding: utf-8 -*-
import sys,subprocess,re

def test_license_plate_recognition():
  print ("Note to self: Increase Bryan's Salary")
  tstDir = "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes"
  cmdStr = "/wrk/acceleration/models/deephi/License_Plate_Recognition_INT8_models_test_codes/run.sh license_plate_recognition_quantizations xfdnn_predict_plate_with_image.py"
  output = ""
  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
  for line in iter(process.stdout.readline, ''): 
    if "jpg" in line:
      output += line
    sys.stdout.write(line)
  process.stdout.close()
  ret = process.wait()
  if ret:
    raise ValueError("License Plate Recognition Deephi Model Fails: crash")

  expected = \
"""
9006.jpg :  津LQ7105 : blue
9012.jpg :  新F50181 : blue
9013.jpg :  新F51298 : blue
902.jpg :  冀C67550 : yellow
9021.jpg :  新F71106 : blue
9024.jpg :  新F89385 : blue
9034.jpg :  新GYU999 : blue
9038.jpg :  新K28586 : blue
9043.jpg :  新A84585 : blue
9052.jpg :  新AF1533 : blue
9054.jpg :  新AJ0418 : blue
9055.jpg :  新AJ7070 : blue
9070.jpg :  新B36480 : blue
9081.jpg :  新C14428 : blue
9094.jpg :  新C26038 : blue
9097.jpg :  新C29737 : blue
"""
  expected = expected.strip("\n")
  output = output.strip("\n")
  
  if output != expected:
    raise ValueError("License Plate Recognition Deephi Model Fails: output invalid")


def test_car_logo_recognition():
  print ("Note to self: Increase Bryan's Salary")
  tstDir = "/wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes"
  cmdStr = "/wrk/acceleration/models/deephi/Car_Logo_Recognition_INT8_models_test_codes/run.sh car_logo_recognition_quantizations xfdnn_predict_car_logo_with_images.py"
  output = ""
  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
  for line in iter(process.stdout.readline, ''): 
    if "jpg" in line:
      output += line
    sys.stdout.write(line)
  process.stdout.close()
  ret = process.wait()
  if ret:
    raise ValueError("Car Logo Recognition Deephi Model Fails: crash")

  expected = \
"""
0cc857585a95.jpg  :  Cadillac
1e7907e660a8.jpg  :  Buick
2fa7fc87b67d.jpg  :  Dongfeng
301ad6277ab2.jpg  :  Mitsubishi
3085c53f6a03.jpg  :  Citroen
401f7163a39f.jpg  :  Mercedes-Benz
60c989c20e37.jpg  :  Buick
78ad7f91466f.jpg  :  Volkswagen
79fd2f689a7d.jpg  :  Nissan
7dba201f8689.jpg  :  Chevrolet
80518ada4262.jpg  :  Fiat
8a47f9831a72.jpg  :  Lexus
9d8c2e9f2967.jpg  :  Volkswagen
a3a2f361964f.jpg  :  Subaru
afbdb4e1a76c.jpg  :  Fiat
c943be91e93d.jpg  :  Great Wall
cbc4af43bb00.jpg  :  Honda
cbdb83d72698.jpg  :  Volkswagen
ddb3b1ea2b3e.jpg  :  Dongfeng
f36be34c0baf.jpg  :  HYUNDAI
"""
  # Below is CPU run for reference
  #0cc857585a95.jpg  :  Cadillac
  #1e7907e660a8.jpg  :  Ford
  #2fa7fc87b67d.jpg  :  Dongfeng
  #301ad6277ab2.jpg  :  Mitsubishi
  #3085c53f6a03.jpg  :  Citroen
  #401f7163a39f.jpg  :  Mercedes-Benz
  #60c989c20e37.jpg  :  Buick
  #78ad7f91466f.jpg  :  Volkswagen
  #79fd2f689a7d.jpg  :  Nissan
  #7dba201f8689.jpg  :  Chevrolet
  #80518ada4262.jpg  :  Fiat
  #8a47f9831a72.jpg  :  Lexus
  #9d8c2e9f2967.jpg  :  Volkswagen
  #a3a2f361964f.jpg  :  Subaru
  #afbdb4e1a76c.jpg  :  Fiat
  #c943be91e93d.jpg  :  Great Wall
  #cbc4af43bb00.jpg  :  Honda
  #cbdb83d72698.jpg  :  Volkswagen
  #ddb3b1ea2b3e.jpg  :  Dongfeng
  #f36be34c0baf.jpg  :  HYUNDAI

  expected = expected.strip("\n")
  output = output.strip("\n")
  
  if output != expected:
    raise ValueError("Car Logo Recognition Deephi Model Fails: output invalid")


def test_car_attributes_recognition():
  print ("Note to self: Increase Bryan's Salary")
  tstDir = "/wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes"
  cmdStr = "/wrk/acceleration/models/deephi/Car_Attributes_Recognition_INT8_models_test_codes/run.sh car_attributes_recognition_quantizations xfdnn_predict_car_attributes_with_images.py"
  output = ""
  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
  for line in iter(process.stdout.readline, ''): 
    if "jpg" in line:
      output += line
    sys.stdout.write(line)
  process.stdout.close()
  ret = process.wait()
  if ret:
    raise ValueError("Car Attributes Recognition Deephi Model Fails: crash")

  expected = \
"""
11_101-161_6763_5_.jpg  type:  minivans  color:  white
12_0-40_1839_1_.jpg  type:  medium and large trucks  color:  blue
26_2744_11.jpg  type:  medium and large trucks  color:  red
28_3570_0_.jpg  type:  sedan  color:  white
car_type_1_4930_0.jpg  type:  sedan  color:  black
car_type_2_5737_0.jpg  type:  medium and large trucks  color:  red
car_type_2_6216_0.jpg  type:  sedan  color:  black
car_type_2_813_0.jpg  type:  sedan  color:  green
ch01_00000000005016700_4.jpg  type:  SUV  color:  white
ch01_00000000008020900_13.jpg  type:  sedan  color:  black
ch01_00000000020000000_2340_9.jpg  type:  sedan  color:  white
ch01_00000000023000000_3030_15.jpg  type:  sedan  color:  black
dongnanlukou_1_36_14190_13.jpg  type:  SUV  color:  white
dongnanlukou_1_36_15510_7.jpg  type:  SUV  color:  white
dongnanlukou_1_37_5730_12.jpg  type:  sedan  color:  silver
sixinyiyuan_tongdao_10_19980_1.jpg  type:  sedan  color:  red
sixinyiyuan_tongdao_18_7620_0.jpg  type:  sedan  color:  blue
tianxiulukou_3_50_10740_8.jpg  type:  sedan  color:  silver
tianxiulukou_3_53_960_6.jpg  type:  sedan  color:  red
"""
  # CPU Output
  #11_101-161_6763_5_.jpg  type:  minivans  color:  white
  #12_0-40_1839_1_.jpg  type:  medium and large trucks  color:  blue
  #26_2744_11.jpg  type:  medium and large trucks  color:  red
  #28_3570_0_.jpg  type:  sedan  color:  white
  #car_type_1_4930_0.jpg  type:  sedan  color:  black
  #car_type_2_5737_0.jpg  type:  medium and large trucks  color:  red
  #car_type_2_6216_0.jpg  type:  sedan  color:  black
  #car_type_2_813_0.jpg  type:  sedan  color:  green
  #ch01_00000000005016700_4.jpg  type:  SUV  color:  white
  #ch01_00000000008020900_13.jpg  type:  sedan  color:  black
  #ch01_00000000020000000_2340_9.jpg  type:  sedan  color:  white
  #ch01_00000000023000000_3030_15.jpg  type:  sedan  color:  black
  #dongnanlukou_1_36_14190_13.jpg  type:  SUV  color:  white
  #dongnanlukou_1_36_15510_7.jpg  type:  SUV  color:  white
  #dongnanlukou_1_37_5730_12.jpg  type:  sedan  color:  silver
  #sixinyiyuan_tongdao_10_19980_1.jpg  type:  sedan  color:  red
  #sixinyiyuan_tongdao_18_7620_0.jpg  type:  sedan  color:  blue
  #tianxiulukou_3_50_10740_8.jpg  type:  sedan  color:  silver
  #tianxiulukou_3_53_960_6.jpg  type:  sedan  color:  red

  expected = expected.strip("\n")
  output = output.strip("\n")
  
  if output != expected:
    raise ValueError("Car Attributes Recognition Deephi Model Fails: output invalid")


def test_pedestrian_attributes_recognition():
  print ("Note to self: Increase Bryan's Salary")
  tstDir = "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes"
  cmdStr = "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition_INT8_models_test_codes/run.sh pedestrian_attributes_recognition_quantizations xfdnn_predict_pedestrian_attributes.py"
  output = ""
  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
  for line in iter(process.stdout.readline, ''): 
    if "bmp" in line:
      output += line
    sys.stdout.write(line)
  process.stdout.close()
  ret = process.wait()
  if ret:
    raise ValueError("Pedestrian Attributes Recognition Deephi Model Fails: crash")

  expected = \
"""
94_790_FRAME_43_RGB.bmp :  {'upper': 'Black', 'lower': 'Blue', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
94_792_FRAME_135_RGB.bmp :  {'upper': 'Black', 'lower': 'Blue', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
94_792_FRAME_43_RGB.bmp :  {'upper': 'Black', 'lower': 'Blue', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
95_798_FRAME_25_RGB.bmp :  {'upper': 'Grey', 'lower': 'Grey', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
95_798_FRAME_46_RGB.bmp :  {'upper': 'Grey', 'lower': 'Grey', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
96_812_FRAME_109_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
96_812_FRAME_53_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
96_815_FRAME_16_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
96_815_FRAME_30_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
97_826_FRAME_113_RGB.bmp :  {'upper': 'White', 'lower': 'Black', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
97_826_FRAME_170_RGB.bmp :  {'upper': 'White', 'lower': 'Black', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
97_829_FRAME_189_RGB.bmp :  {'upper': 'White', 'lower': 'Grey', 'gender': 'Female', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
97_829_FRAME_45_RGB.bmp :  {'upper': 'White', 'lower': 'Black', 'gender': 'Female', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
98_843_FRAME_102_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
98_844_FRAME_12_RGB.bmp :  {'upper': 'Black', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
98_844_FRAME_52_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
98_845_FRAME_158_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
98_845_FRAME_193_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
98_845_FRAME_99_RGB.bmp :  {'upper': 'Grey', 'lower': 'White', 'gender': 'Male', 'backpack': 'Yes', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
99_847_FRAME_111_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
99_847_FRAME_143_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
99_850_FRAME_26_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
99_854_FRAME_22_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
99_854_FRAME_41_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
"""
  # CPU OUTPUT 
  #94_790_FRAME_43_RGB.bmp :  {'upper': 'Black', 'lower': 'Blue', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
  #94_792_FRAME_135_RGB.bmp :  {'upper': 'Black', 'lower': 'Blue', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
  #94_792_FRAME_43_RGB.bmp :  {'upper': 'Black', 'lower': 'Blue', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
  #95_798_FRAME_25_RGB.bmp :  {'upper': 'Grey', 'lower': 'Grey', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
  #95_798_FRAME_46_RGB.bmp :  {'upper': 'Grey', 'lower': 'Grey', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
  #96_812_FRAME_109_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #96_812_FRAME_53_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #96_815_FRAME_16_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #96_815_FRAME_30_RGB.bmp :  {'upper': 'Orange', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #97_826_FRAME_113_RGB.bmp :  {'upper': 'White', 'lower': 'Black', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #97_826_FRAME_170_RGB.bmp :  {'upper': 'White', 'lower': 'Black', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #97_829_FRAME_189_RGB.bmp :  {'upper': 'White', 'lower': 'Grey', 'gender': 'Female', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #97_829_FRAME_45_RGB.bmp :  {'upper': 'White', 'lower': 'Black', 'gender': 'Female', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #98_843_FRAME_102_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #98_844_FRAME_12_RGB.bmp :  {'upper': 'Grey', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
  #98_844_FRAME_52_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #98_845_FRAME_158_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #98_845_FRAME_193_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #98_845_FRAME_99_RGB.bmp :  {'upper': 'Grey', 'lower': 'White', 'gender': 'Male', 'backpack': 'Yes', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #99_847_FRAME_111_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #99_847_FRAME_143_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #99_850_FRAME_26_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #99_854_FRAME_22_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
  #99_854_FRAME_41_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}

  expected = expected.strip("\n")
  output = output.strip("\n")
  
  if output != expected:
    raise ValueError("Pedestrian Attributes Recognition Deephi Model Fails: output invalid")

#  def test_person_reidentification():
#    print ("Note to self: Increase Bryan's Salary")
#    tstDir = "/wrk/acceleration/models/deephi/reid_model_release_20190301"
#    cmdStr = "/wrk/acceleration/models/deephi/reid_model_release_20190301/run.sh deploy xfdnn_reid-inference.py"
#    output = ""
#    capture = False
#    process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#    for line in iter(process.stdout.readline, ''): 
#      if capture:
#        output += line
#      if "Distance matrix" in line:
#        capture = True
#      sys.stdout.write(line)
#    process.stdout.close()
#    ret = process.wait()
#    if ret:
#      raise ValueError("Person Reidentification Deephi Model Fails: crash")
#
#    expected = \
#  """
#  [[ 0.    0.33  0.13  0.41  0.18]
#   [ 0.33  0.    0.26  0.23  0.34]
#   [ 0.13  0.26 -0.    0.36  0.17]
#   [ 0.41  0.23  0.36  0.    0.26]
#   [ 0.18  0.34  0.17  0.26 -0.  ]]
#  Over -------------------
#  """
#    # CPU OUTPUT - quite different - not sure whats up
#    #[[-0.    0.45  0.42  0.49  0.42]
#    # [ 0.45  0.    0.12  0.47  0.39]
#    # [ 0.42  0.12  0.    0.46  0.39]
#    # [ 0.49  0.47  0.46  0.    0.22]
#    # [ 0.42  0.39  0.39  0.22 -0.  ]]
#    # Over -------------------
#
#    expected = expected.strip("\n")
#    output = output.strip("\n")
#    
#    if output != expected:
#      raise ValueError("Person Reidentification Deephi Model Fails: output invalid")

