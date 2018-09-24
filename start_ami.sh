#!/usr/bin/env bash

conda activate ml-suite

. /home/centos/ml-suite/overlaybins/setup.sh aws

/home/centos/anaconda2/envs/ml-suite/bin/jupyter-notebook --config=/home/centos/.jupyter/jupyter_notebook_config.py
