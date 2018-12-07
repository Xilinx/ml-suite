# Installing Anaconda2
Xilinx recommends using Anaconda2 to operate in a virtual environment:
1.  Download Anaconda2  
`wget https://repo.anaconda.com/archive/Anaconda2-5.1.0-Linux-x86_64.sh`
2.  Run the installer (Installer requires bzip, please install it if you don't have it)  
`bash ./Anaconda2-5.1.0-Linux-x86_64.sh`
3.  Ensure that your .bashrc is preparing Anaconda, by including these lines  
      `~/.bashrc: export PATH=/home/<user>/anaconda2/bin:$PATH`  
      `~/.bashrc: . /home/<user>/anaconda2/etc/profile.d/conda.sh`
4.  After updating the bashrc source it to load the new anaconda path  
`source ~/.bashrc`
5.  As a precaution unset PYTHONPATH to avoid conflicts with packages on your rootfs  
`unset PYTHONPATH`

# Create ml-suite Anaconda Virtual Environment
1.  Invoke bash (if you aren't already in bash)  
`bash`
2.  Create Virtual Environment  
`conda create --name ml-suite python=2.7 numpy=1.14.5 x264=20131218 caffe pydot pydot-ng graphviz keras scikit-learn tqdm -c conda-forge`   
3.  Fix symbolic links between pre-compiled Caffe (libcaffe.so), and OpenCV   
      - Note: If you installed anaconda or ml-suite in a custom location, you will need to adjust this. 
      `cd ~/ml-suite/; bash fix_caffe_opencv_symlink.sh`
4.  Activate Environment   
`conda activate ml-suite`
5.  Verify your environment by importing caffe in python  
`python -c "import caffe"`
6.  Install other required python packages
`pip install jupyter tensorflow==1.8 zmq`

### Notes for running on AWS w/ Anaconda
At the moment, AWS requires root privileges to deploy on FPGAs. Follow the steps below as a work around:
 
1)  Become root `sudo su` 
2)  Set Environment Variables Required by runtime `source <MLSUITE_ROOT>/overlaybins/setup.sh aws`
3)  Set User Environment Variables Required to run Anaconda `source ~centos/.bashrc`
4)  Activate the users Anaconda Virtual Environment`source activate ml-suite` 
5)  The environment is setup and ready to run applications and examples.    

### Notes for running on NIMBIX w/ Anaconda
 
1)  Set User Environment Variables Required to run Anaconda `source ~/.bashrc`
2)  Activate the users Anaconda Virtual Environment`source activate ml-suite` 
3)  The environment is setup and ready to run applications and examples.    
