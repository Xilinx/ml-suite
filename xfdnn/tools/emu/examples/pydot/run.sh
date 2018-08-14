#!/bin/bash

#export XDNN_QUANTIZE_CFGFILE=sf_quantize.json
. ../../../../../overlaybins/setup.sh 1525

#python run.py --doemu=True
#python run.py --networkfile=/wrk/hdstaff/satyakee/tf_hub/sample_code/resnet_v2_50_output.pb --pngfile=resnet_v2_50_retrained.png --doemu=True
python run.py --networkfile=/wrk/hdstaff/satyakee/tf_hub/sample_code/resnet_v2_50_output.pb --pngfile=resnet_v2_50_retrained.png
#python run.py --networkfile=/wrk/hdstaff/satyakee/tf_hub/sample_code/output.pb --pngfile=googlenet_v3_retrained.png --doemu=True 
#python run.py --finalnode=final_result
#python run.py --finalnode=final_result
