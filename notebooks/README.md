# Jupyter Notebooks
The Jupyter Notebooks provide easy to use tutorials on how to use the xfDNN tools and deploy models within the Xilinx ML Suite.  

Note for AWS users: The ml-suite AMI on the AWS marketplace already has Jupyter installed, and it will auto-start the notebooks shortly after you launch the EC2 instance. You just need to navigate via a web-browser to \<public-dns\>:8888

## Installation 
1. Install Jupyter on the remote system (We recommend doing this inside your Anaconda environment). 
  ```
  $ source ~/.bashrc
  $ source activate ml-suite
  $ pip install jupyter
  ```
  Follow the instructions to complete the install. You do not need to install Microsoft VScode, when prompted. 
  
2. Navigate to the top level `ml-suite' dir and source the setup script
  ```
  $ cd /ml-suite/
  $ source ./overlaybins/setup.sh <platform>
  ```
  Options for `<platform>`: `aws` `nimbix` `1525` `alveo-u200` `alveo-u250` `alveo-u200-ml` `alveo-u250-ml`

3. Set initial Password for Jupyter Server 
  ```
  $ jupyter notebook --generate-config
  $ jupyter notebook password 
  $ Enter Password: 
  $ Verify Password: 
  $ [NotebookPasswordApp] Wrote hashed password to /root/.jupyter/jupyter_notebook_config.json
  ```

## Launching Notebook Server 

1. Launch the jupyter notebook server  
  `jupyter notebook --no-browser --ip=*`
  
2. On local machine, open a broswer and navigate to `http://localhost:8888` or `youripaddress>:8888`

3. On a remote machine, open a broswer on your local machine navigate to `yourpublicipaddress>:8888`
