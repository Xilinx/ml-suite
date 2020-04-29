import os,subprocess

modelsDir = "/opt/models/caffe/"

models = [
  # "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.bvlc_googlenet_2019-05-02_12_32.zip",
  #  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v2_2019-05-02_12_32.zip",
  #  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v3_2019-05-02_12_32.zip",
  #  "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v4_2019-05-02_12_32.zip",
   "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.resnet50_v1_2019-05-02_12_32.zip",
   # "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.resnet50_v2_2019-05-02_12_32.zip",
   # "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.squeezenet_2019-05-02_12_32.zip",
   # "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.vgg16_2019-05-02_12_32.zip",
   # "https://www.xilinx.com/bin/public/openDownload?filename=models.container.caffe.inception_v2_ssd_2019-05-06_0765.zip",
]

for model in models:
  subprocess.call(["wget",model,"-O","temp.zip"])
  subprocess.call(["unzip","-o","temp.zip"])
  # Strip Unnecessary heirarchy
  for Dir,SubDirs,Files in os.walk("models"):
    if len(Files) > 0:
      pathToParent = Dir
      break
  subprocess.call(["mv",pathToParent,modelsDir])   
  subprocess.call(["rm","-rf","temp.zip","models"])
