
import os

# User can choose to run YOLO at different input sizes
# - 608x608
# - 224x224
# User can choose to run YOLO at different quantization precisions
# - 16b
# - 8b
configs = {
   '608_16b': 
            {'dims': [3, 608, 608], 
             'bitwidths': [16, 16, 16], 
             'network_file': 'yolo_deploy_608.prototxt', 
             'netcfg': 'yolo.cmds', 
             'quantizecfg': 'yolo_deploy_608.json'}, 
   '608_8b': {'dims': [3, 608, 608], 
              'bitwidths': [8, 8, 8], 
              'network_file': 'yolo_deploy_608.prototxt', 
              'netcfg': 'yolo.cmds', 
              'quantizecfg': 'yolo_deploy_608.json'}, 
   '224_16b': {'dims': [3, 224, 224], 
               'bitwidths': [16, 16, 16], 
               'network_file': 'yolo_deploy_224.prototxt', 
               'netcfg': 'yolo.cmds', 
               'quantizecfg': 'yolo_deploy_224.json'}, 
   '224_8b': {'dims': [3, 224, 224], 
              'bitwidths': [8, 8, 8], 
              'network_file': 'yolo_deploy_224.prototxt', 
              'netcfg': 'yolo.cmds', 
              'quantizecfg': 'yolo_deploy_224.json'}}

# Choose a config here
config = configs['608_16b']
#config = configs['608_8b']
#config = configs['224_16b']
#config = configs['224_8b']

aws = False
eb = False
if config['bitwidths'][0] == 8:
    eb = True
if os.path.exists('/sys/hypervisor/uuid'):
    with open('/sys/hypervisor/uuid') as (file):
        contents = file.read()
        if 'ec2' in contents:
            print 'Runnning on Amazon AWS EC2'
            aws = True
            if eb:
                xclbin = '../../overlaybins/aws/xdnn_56_8b_5m.awsxclbin'
            else:
                xclbin = '../../overlaybins/aws/xdnn_56_16b_5m.awsxclbin'
if not aws:
    if eb:
        xclbin = '../../overlaybins/1525/xdnn_56_8b_5m.xclbin'
    else:
        xclbin = '../../overlaybins/1525/xdnn_56_16b_5m.xclbin'
