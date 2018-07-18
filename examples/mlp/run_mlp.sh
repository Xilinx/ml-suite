#/bin/bash
source rt_setup.sh xclbins/$1
python examples/keras/mlp.py --data ./examples/keras/data/SansEC_Train_Data.csv --model examples/keras/best_model.h5 --xclbin xclbins/$1/gemx.xclbin --cfg xclbins/$1/config_info.dat --gemxlib ./out_host/lib/libgemxhost.so
