# Create ml-suite Anaconda Virtual Environment
1.  Invoke bash (if you aren't already in bash)  
`bash`
2.  Create Virtual Environment  
`conda create --name ml-suite python=2.7 caffe pydot pydot-ng graphviz -c conda-forge`
3.  Fix symbolic links between pre-compiled Caffe (libcaffe.so), and OpenCV   
`bash fix_caffe_opencv_symlink.sh`
4.  Activate Environment   
`conda activate ml-suite`
5.  Verify your environment by importing caffe in python  
`python -c "import caffe"`
6. Deactivate Enviorment
`conda deactivate `


### Notes for running on AWS w/ Anaconda
 AWS requires root priveleges to deploy on FPGAs, which makes Anaconda usage difficult. Follow the steps belowas a work around, until a better process is found
 
1)  Become Root `sudo su` 
2)  Set Environment Variables Required by runtime `source */overlaybins/aws/setup.sh`
3)  Set User Environment Variables Required to run Anaconda `source ~centos/.bashrc`
4)  Activate the users Anaconda Virtual Environment`source activate Caffe27` 
5)  Finally, run the script we wish to run `python test.py`  

