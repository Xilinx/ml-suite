# Jupyter Notebooks
The Jupyter Notebooks provide tutorials on how to run models within the Xilinx ML Suite.  
Jupyter is preinstalled in the Xilinx ML Suite Docker image.

## Notebook Setup
Follow these instructions from inside a running container
1. Install the necessary dataset
  ```
  # For Imagenet
  cd /opt/ml-suite/examples/caffe
  python -m ck pull repo:ck-env
  python -m ck install package:imagenet-2012-val-min
  python -m ck install package:imagenet-2012-aux
  head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > \
  $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
  # Resize all the images to a common dimension for Caffe
  python resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
  # Get the necessary models
  python getModels.py
  ```
  
2. Launch Jupyter notebook server
  ```
  cd /opt/ml-suite/notebooks
  jupyter notebook --no-browser --ip=0.0.0.0 --NotebookApp.token='' --NotebookApp.password=''
  ```
  
3. Open a broswer, and navigate to one of:  
  - `<yourpublicipaddress>:8888`
  - `<yourdns>:8888`
  - `<yourhostname>:8888`
