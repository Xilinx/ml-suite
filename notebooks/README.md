# Jupyter Notebooks
The Jupyter Notebooks in this directory give overviews of various software components contained within the Xilinx ML Suite.

## Setup
If you are running on a remote system such as Amazon AWS, you will need to communicate with Jupyter Notebook server over ssh.
If you are running localy, you can skip the first step.  
If you already have jupyter installed on the remote machine, you don't have to install it again.  
1. SSH to the remote machine with local port forwarding  
  * `ssh -i my_pem_key.pem -L 8889:localhost:8888 10.22.64.108`
  * You will need to replace the key name, and ip address above with your systems parameters  
2. Install Jupyter on the remote system (We recommend doing this inside your Anaconda environment).  
  * `bash`
  * `conda activate ml-suite`
  * `pip install jupyter`
3. Source the setup script we have provied for your hardware in overlaybins  
  * `cd */ml-suite`
  * `source ./overlaybins/setup.sh <platform>`
  * You may use aws or 1525 for the <platform> argument, depending on your available hardware
4. Launch the jupyter notebook server  
  * `jupyter notebook --no-browser
   
  
At this point, you should be able to access your notebook server on your local machine using the url http://localhost:8889

If you are running locally without the local port forwarding, you can use http://localhost:8888

You may run into password/token issues when logging in for the first time. Just fight through it, and you'll get there...
  
