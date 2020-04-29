# Based on caffe example

cd /opt/ml-suite/examples/caffe || exit
python -m ck pull repo:ck-env
python -m ck install package:imagenet-2012-val-min
python -m ck install package:imagenet-2012-aux
head -n 500 $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-aux/val.txt > $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min/val_map.txt
python resize.py $HOME/CK-TOOLS/dataset-imagenet-ilsvrc2012-val-min 256 256
python getModels.py
source $MLSUITE_ROOT/overlaybins/setup.sh

# Quantize
python run.py --prototxt /opt/models/caffe/resnet50_v1/resnet50_v1_train_val.prototxt --caffemodel /opt/models/caffe/resnet50_v1/resnet50_v1.caffemodel --prepare --output_dir work

# Install gRPC
python -m pip install grpcio-tools