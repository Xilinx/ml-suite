# ImageNet Image Classification w/ GoogLeNet v1 Web Demo

## Introduction
This tutorial shows you how to launch an image classification GoogLeNet-v1 demo in a web browser.  
Once the demo is started, you will be able to view the demo and monitor performance from any web browser.
 
This demo can be ran on any hardware platform.

For instructions on launching and connecting to an aws instances, see [here][].
  
1. Ensure you have an Anaconda environment setup  
  - [Setup Anaconda][]

2. Download ImageNet Validation set  
    This demo is meant to use ImageNet ILSVRC2012 validation files.  
    You need to download the files and store them [here.](models/data/ilsvrc12/ilsvrc12_img_val)  
    The following naming convention: ILSVRC2012_val_<IMAGE ID>.JPEG where <IMAGE_ID> starts at 00000000.  Instructions for downloading ILSVRC2012 files can be found here: http://www.image-net.org/download-imageurls

    If you want to try it without downloading the dataset, you can insert you own files and simply name them:
    - ILSVRC2012_val_00000000.JPEG
    - ILSVRC2012_val_00000001.JPEG
    - ILSVRC2012_val_00000002.JPEG
    - etc.

    The demo should start as long as there is at least one image to classify. Note that the demo webpage shows the correct answer above each image and that is indexed by the ID of the ILSVRC2012 images, so if the images provided are not part of the ILSVRC2012 dataset, there may be a mismatch in the reported labels.

    Download and extract the files to the following locations:
    ml-suite/apps/perpetual_demo/www/imagenet_val (for Terminal 1 below)  
    ml-suite/examples/classification/imagenet_val (for Terminal 2 below)  

3. Terminal 1: run Web GUI  
    ```sh
    $ conda activate ml-suite
    $ cd ml-suite/apps/perpetual_demo
    ```
    
4. Execute `run.sh` script
    ```sh
    $ ./run.sh
    ```
    This starts the web GUI @ http://<your_ip>:8998/static/www/index.html  
    Note: you may need to open port 8998, if it is not already open. For AWS see the console settings.  
    
5. Open a new Terminal (Terminal 2) and start anaconda environment and navigate to the Classification example
    ```sh
    # Recall that if you are running on AWS, you need to first become root user, thanks to their sudo requirement.
    # sudo su; source ~centos/.bashrc
    $ conda activate ml-suite
    $ cd ml-suite/examples/classification
    ```
    
6. Execute `run.sh` script with the following arguments
    ```
    $ ./run.sh -p alveo-u200 -t perpetual
    
    # Remember the first argument can be aws, nimbix, 1525, etc... refer to documentation in /examples/classification
    ```

8. Open Web Browser  
    Navigate to `http://<your_ip>:8998/static/www/index.html`

![](../../docs/tutorials/img/image_classification.png)



[here]: ../../docs/tutorials/aws-f1-launching.md
[Setup Anaconda]: ../../docs/tutorials/anaconda.md
