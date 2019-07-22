import os, subprocess, shutil

modelsDir = "/opt/models/tensorflow/"

models = [
  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.tensorflow.inception_v1_baseline.pb_2019-07-18.zip",
  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.tensorflow.resnet50_baseline.pb_2019-07-18.zip"
]

for model in models:
  subprocess.call(["wget", model, "-O", "temp.zip"])
  subprocess.call(["unzip", "-o", "temp.zip"])
  for Dir,SubDirs,Files in os.walk("models"):
    for file_name in Files:
      full_file_name = os.path.join(Dir, file_name)
      if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, modelsDir)
subprocess.call(["rm", "-rf", "temp.zip", "models"])
