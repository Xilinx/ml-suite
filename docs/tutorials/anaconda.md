# Installing Anaconda2
Xilinx recommends using Anaconda2 to operate in a virtual environment:
1.  Download Anaconda2  
`wget https://repo.anaconda.com/archive/Anaconda2-5.1.0-Linux-x86_64.sh`
2.  Run the installer  
`bash ./Anaconda2-5.1.0-Linux-x86_64.sh`
3.  Ensure that your .bashrc is preparing Anaconda, by including these lines  
      `~/.bashrc: export PATH=/home/<user>/anaconda2/bin:$PATH`  
      `~/.bashrc: . /home/<user>/anaconda2/etc/profile.d/conda.sh`
4.  After updating the bashrc source it to load the new anaconda path  
`source ~/.bashrc`
5.  As a precaution unset PYTHONPATH to avoid conflicts with packages on your rootfs  
`unset PYTHONPATH`
