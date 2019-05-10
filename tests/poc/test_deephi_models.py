#!/usr/bin/env python
#
# // SPDX-License-Identifier: BSD-3-CLAUSE
#
# (C) Copyright 2018, Xilinx, Inc.
#
## -*- coding: utf-8 -*-
#import sys,subprocess,re
#
#def test_car_attributes_recognition():
#  tstDir = "/wrk/acceleration/models/deephi/Car_Attributes_Recognition"
#  cmdStr = "/wrk/acceleration/models/deephi/Car_Attributes_Recognition/run.sh deploy xfdnn_predict_car_attributes_with_images.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Car Attributes Recognition Deephi Model Fails: crash")
#
#  expected = \
#"""
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
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Car Attributes Recognition Deephi Model Fails: output invalid")
#
#
#def test_car_logo_recognition():
#  tstDir = "/wrk/acceleration/models/deephi/Car_Logo_Recognition"
#  cmdStr = "/wrk/acceleration/models/deephi/Car_Logo_Recognition/run.sh deploy xfdnn_predict_car_logo_with_images.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Car Logo Recognition Deephi Model Fails: crash")
#
#  expected = \
#"""
#0cc857585a95.jpg  :  Cadillac
#1e7907e660a8.jpg  :  Buick
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
#"""
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Car Logo Recognition Deephi Model Fails: output invalid")
#
#
#def test_car_logo_detection():
#  tstDir = "/wrk/acceleration/models/deephi/Car_Logo_Detection1"
#  cmdStr = "/wrk/acceleration/models/deephi/Car_Logo_Detection1/run.sh deploy xfdnn_densebox_detect_car_logo.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Car_Logo_Detection Deephi Model Fails: crash")
#
#  expected = \
#"""
#('car_face_5.jpg', 164, 150, 190, 171, 0.9626731126558705)
#('car_face_6.jpg', 191, 124, 246, 160, 0.991422514586288)
#('car_face_0.jpg', 663, 200, 725, 246, 0.9433475746920261)
#('car_face_1.jpg', 'no object detection!')
#('car_face_2.jpg', 151, 179, 190, 209, 0.9808759632491112)
#('car_face_3.jpg', 190, 222, 242, 248, 0.9808759632491111)
#('car_face_4.jpg', 149, 187, 182, 217, 0.9840936082881853)
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Car Logo Detection Deephi Model Fails: output invalid")
#	
#	
#def test_license_plate_recognition():
#  tstDir = "/wrk/acceleration/models/deephi/License_Plate_Recognition"
#  cmdStr = "/wrk/acceleration/models/deephi/License_Plate_Recognition/run.sh deploy xfdnn_predict_plate_with_image.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("License Plate Recognition Deephi Model Fails: crash")
#
#  expected = \
#"""
#9006.jpg :  津LQ7105 : blue
#9012.jpg :  新F50181 : blue
#9013.jpg :  新F51298 : blue
#902.jpg :  冀C67550 : yellow
#9021.jpg :  新F71106 : blue
#9024.jpg :  新F89385 : blue
#9034.jpg :  新GYU999 : blue
#9038.jpg :  新K28586 : blue
#9043.jpg :  新A84585 : blue
#9052.jpg :  新AF1533 : blue
#9054.jpg :  新AJ0418 : blue
#9055.jpg :  新AJ7070 : blue
#9070.jpg :  新B36480 : blue
#9081.jpg :  新C14428 : blue
#9094.jpg :  新C26038 : blue
#9097.jpg :  新C29737 : blue
#"""
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("License Plate Recognition Deephi Model Fails: output invalid")
#
#
#def test_license_plate_detection():
#  tstDir = "/wrk/acceleration/models/deephi/License_Plate_Detection"
#  cmdStr = "/wrk/acceleration/models/deephi/License_Plate_Detection/run.sh deploy xfdnn_densebox_detect_plate.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("License Plate Detection Deephi Model Fails: crash")
#
#  expected = \
#"""
#('vehicle_0000424_0.jpg', 104, 262, 172, 283, 0.9999251537724894)
#('vehicle_0003000_0.jpg', 134, 280, 193, 294, 0.9947798743064417)
#('vehicle_0003030_0.jpg', 115, 276, 191, 296, 0.9992903296008995)
#('vehicle_0003060_0.jpg', 126, 254, 203, 276, 0.9947798743064417)
#('vehicle_0003090_0.jpg', 128, 277, 205, 296, 0.9999251537724894)
#('vehicle_0003125_0.jpg', 112, 266, 190, 284, 0.9997388096809043)
#('vehicle_0003155_0.jpg', 120, 269, 192, 289, 0.9999417087343387)
#('vehicle_0003185_0.jpg', 151, 272, 235, 290, 0.9999151889582764)
#('vehicle_0003215_0.jpg', 128, 285, 187, 298, 0.9770226300899744)
#"""
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("License Plate Detection Deephi Model Fails: output invalid")
#	
#	
#def test_pedestrian_attributes_recognition():
#  tstDir = "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition"
#  cmdStr = "/wrk/acceleration/models/deephi/Pedestrian_Attributes_Recognition/run.sh deploy xfdnn_predict_pedestrian_attributes.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "bmp" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Pedestrian Attributes Recognition Deephi Model Fails: crash")
#
#  expected = \
#"""
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
#98_844_FRAME_12_RGB.bmp :  {'upper': 'Black', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'Yes', 'handbag': 'No', 'hat': 'No'}
#98_844_FRAME_52_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#98_845_FRAME_158_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#98_845_FRAME_193_RGB.bmp :  {'upper': 'Blue', 'lower': 'White', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#98_845_FRAME_99_RGB.bmp :  {'upper': 'Grey', 'lower': 'White', 'gender': 'Male', 'backpack': 'Yes', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#99_847_FRAME_111_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#99_847_FRAME_143_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#99_850_FRAME_26_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#99_854_FRAME_22_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#99_854_FRAME_41_RGB.bmp :  {'upper': 'White', 'lower': 'Brown', 'gender': 'Male', 'backpack': 'No', 'bag': 'No', 'handbag': 'No', 'hat': 'No'}
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Pedestrian Attributes Recognition Deephi Model Fails: output invalid")
#
#
#def test_pedestrian_detection():
#  tstDir = "/wrk/acceleration/models/deephi/Pedestrian_Detection"
#  cmdStr = "/wrk/acceleration/models/deephi/Pedestrian_Detection/run.sh deploy xfdnn_ssd_like_detect.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Pedestrian Detection Deephi Model Fails: crash")
#
#  expected = \
#"""
#('COCO_val2014_000000003134.jpg', 'no pedestrian detection')
#('COCO_val2014_000000192838.jpg', [u'person', 0.86703575, 274, 154, 467, 299])
#('COCO_val2014_000000000136.jpg', [u'person', 0.936285, 0, 71, 67, 378])
#('COCO_val2014_000000000139.jpg', [u'person', 0.91490096, 422, 155, 464, 303])
#('COCO_val2014_000000000544.jpg', [u'person', 0.98173577, 251, 238, 367, 410])
#('COCO_val2014_000000000544.jpg', [u'person', 0.98173577, 26, 274, 127, 411])
#('COCO_val2014_000000000544.jpg', [u'person', 0.98173577, 101, 302, 236, 404])
#('COCO_val2014_000000000544.jpg', [u'person', 0.9591543, 1, 217, 39, 299])
#('COCO_val2014_000000000544.jpg', [u'person', 0.8267118, 23, 239, 65, 296])
#('COCO_val2014_000000000641.jpg', 'no pedestrian detection')
#('COCO_val2014_000000000692.jpg', [u'person', 0.9433476, 199, 88, 501, 479])
#('COCO_val2014_000000000962.jpg', [u'person', 0.8740772, 30, 169, 361, 606])
#('COCO_val2014_000000001292.jpg', [u'person', 0.8080672, 213, 33, 468, 374])
#('COCO_val2014_000000001404.jpg', 'no pedestrian detection')
#('COCO_val2014_000000001987.jpg', [u'person', 0.98173577, 199, 51, 282, 215])
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Pedestrian Detection Deephi Model Fails: output invalid")
#	
#	
#def test_person_reidentification():
#  tstDir = "/wrk/acceleration/models/deephi/reidentification"
#  cmdStr = "/wrk/acceleration/models/deephi/reidentification/run.sh deploy xfdnn_reid-inference.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Person reidentification Deephi Model Fails: crash")
#
#  expected = \
#"""
#('person_0.jpg', 'person_0.jpg', 0.0)
#('person_0.jpg', 'person_1.jpg', 0.4746243357658386)
#('person_0.jpg', 'person_2.jpg', 0.4006388783454895)
#('person_0.jpg', 'person_3.jpg', 0.32887470722198486)
#('person_0.jpg', 'person_4.jpg', 0.2933369278907776)
#('person_1.jpg', 'person_0.jpg', 0.4746243357658386)
#('person_1.jpg', 'person_1.jpg', 0.0)
#('person_1.jpg', 'person_2.jpg', 0.10194259881973267)
#('person_1.jpg', 'person_3.jpg', 0.39943844079971313)
#('person_1.jpg', 'person_4.jpg', 0.4299378991127014)
#('person_2.jpg', 'person_0.jpg', 0.4006388783454895)
#('person_2.jpg', 'person_1.jpg', 0.10194259881973267)
#('person_2.jpg', 'person_2.jpg', 0.0)
#('person_2.jpg', 'person_3.jpg', 0.34741491079330444)
#('person_2.jpg', 'person_4.jpg', 0.37494170665740967)
#('person_3.jpg', 'person_0.jpg', 0.32887470722198486)
#('person_3.jpg', 'person_1.jpg', 0.39943844079971313)
#('person_3.jpg', 'person_2.jpg', 0.34741491079330444)
#('person_3.jpg', 'person_3.jpg', 0.0)
#('person_3.jpg', 'person_4.jpg', 0.22483551502227783)
#('person_4.jpg', 'person_0.jpg', 0.2933369278907776)
#('person_4.jpg', 'person_1.jpg', 0.4299378991127014)
#('person_4.jpg', 'person_2.jpg', 0.37494170665740967)
#('person_4.jpg', 'person_3.jpg', 0.22483551502227783)
#('person_4.jpg', 'person_4.jpg', 0.0)
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Person reidentification Model Fails: output invalid")
#	
#	
## def test_pose_estimation():
#  # tstDir = "/wrk/acceleration/models/deephi/Pose_Estimation"
#  # cmdStr = "/wrk/acceleration/models/deephi/Pose_Estimation/run.sh deploy xfdnn_pose_estimation.py"
#  # output = ""
#  # process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  # for line in iter(process.stdout.readline, ''): 
#    # if "jpg" in line:
#      # output += line
#    # sys.stdout.write(line)
#  # process.stdout.close()
#  # ret = process.wait()
#  # if ret:
#    # raise ValueError("Pose_Estimation Deephi Model Fails: crash")
#
#  # expected = \
## """
## ('test_7.jpg', 89, 60)
## ('test_7.jpg', 121, 14)
## ('test_7.jpg', 110, 25)
## ('test_7.jpg', 76, 67)
## ('test_7.jpg', 76, 77)
## ('test_7.jpg', 102, 73)
## ('test_7.jpg', 92, 174)
## ('test_7.jpg', 56, 243)
## ('test_7.jpg', 29, 256)
## ('test_7.jpg', 84, 179)
## ('test_7.jpg', 102, 260)
## ('test_7.jpg', 51, 290)
## ('test_7.jpg', 78, 174)
## ('test_7.jpg', 72, 99)
## ('test_1.jpg', 82, 57)
## ('test_1.jpg', 108, 14)
## ('test_1.jpg', 103, 7)
## ('test_1.jpg', 79, 66)
## ('test_1.jpg', 69, 70)
## ('test_1.jpg', 100, 35)
## ('test_1.jpg', 75, 172)
## ('test_1.jpg', 39, 247)
## ('test_1.jpg', 23, 262)
## ('test_1.jpg', 75, 174)
## ('test_1.jpg', 97, 263)
## ('test_1.jpg', 45, 272)
## ('test_1.jpg', 77, 181)
## ('test_1.jpg', 69, 100)
## ('test_2.jpg', 78, 61)
## ('test_2.jpg', 98, 29)
## ('test_2.jpg', 80, 47)
## ('test_2.jpg', 75, 68)
## ('test_2.jpg', 77, 73)
## ('test_2.jpg', 105, 56)
## ('test_2.jpg', 75, 175)
## ('test_2.jpg', 49, 243)
## ('test_2.jpg', 39, 248)
## ('test_2.jpg', 77, 179)
## ('test_2.jpg', 96, 260)
## ('test_2.jpg', 51, 268)
## ('test_2.jpg', 78, 169)
## ('test_2.jpg', 69, 100)
## ('test_3.jpg', 87, 52)
## ('test_3.jpg', 110, 31)
## ('test_3.jpg', 100, 55)
## ('test_3.jpg', 70, 53)
## ('test_3.jpg', 82, 67)
## ('test_3.jpg', 106, 61)
## ('test_3.jpg', 87, 168)
## ('test_3.jpg', 68, 236)
## ('test_3.jpg', 54, 237)
## ('test_3.jpg', 73, 168)
## ('test_3.jpg', 82, 257)
## ('test_3.jpg', 46, 271)
## ('test_3.jpg', 78, 168)
## ('test_3.jpg', 71, 91)
## ('test_4.jpg', 97, 96)
## ('test_4.jpg', 113, 33)
## ('test_4.jpg', 105, 26)
## ('test_4.jpg', 76, 118)
## ('test_4.jpg', 80, 89)
## ('test_4.jpg', 101, 55)
## ('test_4.jpg', 87, 200)
## ('test_4.jpg', 45, 244)
## ('test_4.jpg', 33, 248)
## ('test_4.jpg', 79, 207)
## ('test_4.jpg', 99, 267)
## ('test_4.jpg', 66, 259)
## ('test_4.jpg', 86, 162)
## ('test_4.jpg', 76, 134)
## ('test_5.jpg', 92, 55)
## ('test_5.jpg', 120, 27)
## ('test_5.jpg', 105, 25)
## ('test_5.jpg', 67, 64)
## ('test_5.jpg', 79, 81)
## ('test_5.jpg', 105, 54)
## ('test_5.jpg', 86, 178)
## ('test_5.jpg', 45, 240)
## ('test_5.jpg', 33, 221)
## ('test_5.jpg', 77, 184)
## ('test_5.jpg', 81, 266)
## ('test_5.jpg', 23, 265)
## ('test_5.jpg', 77, 172)
## ('test_5.jpg', 73, 97)
## ('test_6.jpg', 92, 82)
## ('test_6.jpg', 93, 40)
## ('test_6.jpg', 52, 43)
## ('test_6.jpg', 72, 91)
## ('test_6.jpg', 81, 89)
## ('test_6.jpg', 103, 73)
## ('test_6.jpg', 92, 181)
## ('test_6.jpg', 52, 236)
## ('test_6.jpg', 45, 246)
## ('test_6.jpg', 83, 182)
## ('test_6.jpg', 94, 255)
## ('test_6.jpg', 59, 285)
## ('test_6.jpg', 81, 167)
## ('test_6.jpg', 75, 116)
## ('test_0.jpg', 80, 60)
## ('test_0.jpg', 93, 32)
## ('test_0.jpg', 62, 27)
## ('test_0.jpg', 80, 64)
## ('test_0.jpg', 82, 79)
## ('test_0.jpg', 109, 57)
## ('test_0.jpg', 77, 192)
## ('test_0.jpg', 36, 261)
## ('test_0.jpg', 32, 280)
## ('test_0.jpg', 82, 190)
## ('test_0.jpg', 96, 273)
## ('test_0.jpg', 62, 279)
## ('test_0.jpg', 79, 169)
## ('test_0.jpg', 73, 98)
## """
#
#  # expected = expected.strip("\n")
#  # output = output.strip("\n")
#  
#  # if output != expected:
#    # raise ValueError("Pose_Estimation Model Fails: output invalid")
#	
#	
#def test_face_attribute_recognition():
#  tstDir = "/wrk/acceleration/models/deephi/FaceSamples/models/landmark_attr_model"
#  cmdStr = "/wrk/acceleration/models/deephi/FaceSamples/models/landmark_attr_model/run.sh deploy ../../sample_code/landmark_attr_code/xfdnn_landmark_attr_sample.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Face Attribute Recognition Deephi Model Fails: crash")
#
#  expected = \
#"""
#1.jpg GENDER: MALE AGE: 35.564026
#2.jpg GENDER: FEMALE AGE: 37.553238
#3.jpg GENDER: FEMALE AGE: 60.386097
#4.jpg GENDER: MALE AGE: 56.97293
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Face Attribute Recognition Model Fails: output invalid")
#
#
#def test_face_recognition():
#  tstDir = "/wrk/acceleration/models/deephi/FaceSamples/models/recog_model"
#  cmdStr = "/wrk/acceleration/models/deephi/FaceSamples/models/recog_model/run.sh deploy ../../sample_code/recog_code/xfdnn_face_recog_sample.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Face Recognition Deephi Model Fails: crash")
#
#  expected = \
#"""
#['recog_pairs/negative/1/Nova_Esther_Guthrie_0001.jpg', 'recog_pairs/negative/1/Stephen_Joseph_0001.jpg'] 0.379084762433
#['recog_pairs/negative/3/Peter_OToole_0001.jpg', 'recog_pairs/negative/3/Peter_Gabriel_0001.jpg'] 0.304219263423
#['recog_pairs/negative/4/William_Rosenberg_0001.jpg', 'recog_pairs/negative/4/Peter_Lundgren_0001.jpg'] 0.453853590941
#['recog_pairs/negative/5/Randy_Brown_0001.jpg', 'recog_pairs/negative/5/Val_Ackerman_0001.jpg'] 0.167763014275
#['recog_pairs/negative/2/Romain_Duris_0001.jpg', 'recog_pairs/negative/2/Peter_Ahearn_0001.jpg'] 0.0517978928223
#['recog_pairs/positive/1/Charlie_Zaa_0001.jpg', 'recog_pairs/positive/1/Charlie_Zaa_0002.jpg'] 0.942530772978
#['recog_pairs/positive/3/Ed_Smart_0003.jpg', 'recog_pairs/positive/3/Ed_Smart_0001.jpg'] 0.899960311637
#['recog_pairs/positive/4/Guy_Hemmings_0002.jpg', 'recog_pairs/positive/4/Guy_Hemmings_0001.jpg'] 0.813860878045
#['recog_pairs/positive/5/Iva_Majoli_0002.jpg', 'recog_pairs/positive/5/Iva_Majoli_0001.jpg'] 0.991060052661
#['recog_pairs/positive/2/David_Wells_0002.jpg', 'recog_pairs/positive/2/David_Wells_0001.jpg'] 0.871355200479
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Face Recognition Model Fails: output invalid")
#
#
#def test_face_detection():
#  tstDir = "/wrk/acceleration/models/deephi/FaceSamples/models/densebox_model"
#  cmdStr = "/wrk/acceleration/models/deephi/FaceSamples/models/densebox_model/run.sh deploy ../../sample_code/densebox_code/xfdnn_densebox_sample.py"
#  output = ""
#  process = subprocess.Popen(cmdStr,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=True,cwd=tstDir)
#  for line in iter(process.stdout.readline, ''): 
#    if "jpg" in line:
#      output += line
#    sys.stdout.write(line)
#  process.stdout.close()
#  ret = process.wait()
#  if ret:
#    raise ValueError("Face Detection Deephi Model Fails: crash")
#
#  expected = \
#"""
#('img_82.jpg', 127, 82, 192, 178)
#('img_136.jpg', 126, 24, 189, 94)
#('img_136.jpg', 258, 30, 303, 93)
#('img_136.jpg', 197, 22, 227, 67)
#('img_151.jpg', 131, 72, 181, 131)
#('img_158.jpg', 44, 42, 243, 257)
#('img_158.jpg', 118, 73, 206, 175)
#('img_176.jpg', 97, 40, 148, 105)
#('img_230.jpg', 97, 27, 180, 166)
#('img_230.jpg', 258, 97, 300, 150)
#('img_230.jpg', 22, 65, 68, 134)
#('img_230.jpg', 221, 78, 256, 127)
#('img_326.jpg', 79, 53, 211, 231)
#('img_327.jpg', 92, 70, 248, 243)
#('img_331.jpg', 111, 27, 191, 123)
#('img_331.jpg', 41, 70, 109, 155)
#('img_336.jpg', 217, 58, 275, 157)
#('img_336.jpg', 85, 44, 156, 163)
#('img_346.jpg', 166, 57, 230, 141)
#('img_346.jpg', 232, 45, 282, 127)
#('img_346.jpg', 92, 40, 144, 127)
#('img_457.jpg', 139, 47, 209, 184)
#('img_458.jpg', 112, 69, 272, 242)
#('img_475.jpg', 76, 76, 249, 258)
#('img_496.jpg', 104, 32, 196, 147)
#('img_532.jpg', 81, 40, 146, 140)
#('img_532.jpg', 210, 126, 282, 233)
#('img_532.jpg', 37, 117, 79, 170)
#('img_532.jpg', 152, 87, 202, 159)
#('img_532.jpg', 282, 111, 320, 185)
#('img_547.jpg', 92, 39, 223, 194)
#('img_558.jpg', 'no face detection')
#('img_588.jpg', 129, 56, 204, 175)
#('img_698.jpg', 137, 28, 193, 120)
#"""
#
#  expected = expected.strip("\n")
#  output = output.strip("\n")
#  
#  if output != expected:
#    raise ValueError("Face Detection Model Fails: output invalid")
