# ImageNet Image Classification w/ GoogLeNet v1 Web Demo

## Introduction
This tutorial shows you how to launch an image classification GoogLeNet-v1 demo in a web browser.  
Once the demo is started, you will be able to view the demo and monitor performance from any web browser.
 
This demo can be run on AWS or a local machine with a VCU1525 card.

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
4. Open `run.sh` and add or remove the number of ports according to the number of FPGAs on the machine starting from 5505 through 55x5 for CPP listeners and 5506 through 55x6 for XMLRT listeners. For example if there are 4 FPGAs:
    ```
    python server.py -z 127.0.0.1:5505,5515,5525,5535 -x 127.0.0.1:5506,5516,5526,5536
    ```
     *Note that no changes are necessary for a single FPGA.

5. Execute `run.sh` script
    ```sh
    $ ./run.sh
    ```
    This starts the web GUI @ http://<your_ip>:8998/static/www/index.html  
    Note: you may need to open port 8998, if it is not already open. For AWS see the console settings.  
    
6. Open a new Terminal (Terminal 2) and start anaconda environment and navigate to the Classification examples
    ```sh
    $ cd ml-suite/examples/classification
    ```
7. Open `run_perpetual_demo.sh` and edit the for loop to run all the FPGAs. By default, all of them run Googlenet V1, for other networks, add the flag -m <model name>. Note that while most platforms can be autodetected by the software, some of them can't. You may need to edit the command to provide -p <platform name>
    ```
    for i in {0..0} ;
    do
        unset PYTHONPATH
        ./run.sh -t streaming_classify -m resnet50 -k v3 -b 8 -i $i -x -v > /dev/null & 
    done
    ```

8. Execte `run_perpetual_demo.sh`
    ```sh
    $ ./run_perpetual_demo.sh 
    ```

9. Open Web Browser  
    Navigate to `http://<your_ip>:8998/static/www/index.html`
    If there are multiple FPGAS, Navigate to `http://<your_ip>:8998/static/www/index.html#<no_of_FPGAS>`
    like `http://<your_ip>:8998/static/www/index.html#4`

![](../../docs/tutorials/img/image_classification.png)



[here]: ../../docs/tutorials/aws-f1-launching.md
[Setup Anaconda]: ../../docs/tutorials/anaconda.md
