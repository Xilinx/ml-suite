# emDNN OneShot - ResNet

This folder contains caffe network files (prototxt and caffemodel) for classification of images (32x32x3), using ResNet18 and ResNet18 optimized using emDNN, from CIFAR-10 dataset.

### Model Description
* resnet18_baseline.caffemodel  -> Weights : ResNet18 (CIFAR-10) CNN (baseline)
* resnet18_baseline_deploy.prototxt    -> Network : ResNet18 (CIFAR-10) CNN (baseline)
* resnet18_emdnn.caffemodel     -> Weights : ResNet18_emDNN (CIFAR-10) based on ResNet18 Baseline
* resnet18_emdnn_deploy.prototxt       -> Network : ResNet18_emDNN (CIFAR-10) based on ResNet18 Baseline

## Executing Evaluation

### Requirements
* Caffe 1.1.0
* numpy 1.14.3
* python2.7

### Execution
```bash
python eval_model.py --model_def=<model_deploy.prototxt> --model_weights=<model.caffemodel> --npz_data=<data.npz> --batch_size=<batch_size>'
```

where
* ```model_def``` is path to prototxt file of model
* ```model_weights``` is path to caffemodel file of model
* ```npz_data``` is path to cifar 10 normalized data saved in numpy compressed
* ```batch_size``` number of images to processed in a batch

### Examples

To execute baseline model :
```bash
python eval_model.py --model_def=resnet18_baseline_deploy.prototxt --model_weights=resnet18_baseline.caffemodel --npz_data=cifar_normalized.npz --batch_size=100
```

To execute emdnn optimized model :
```bash
python eval_model.py --model_def=resnet18_emdnn_deploy.prototxt --model_weights=resnet18_emdnn.caffemodel --npz_data=cifar_normalized.npz --batch_size=100
```
