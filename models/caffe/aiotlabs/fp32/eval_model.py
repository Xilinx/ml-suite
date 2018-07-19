import os
os.environ['GLOG_minloglevel'] = '2'
import time
import sys
import getopt

import caffe
import numpy as np

# Everything runs on single thread
def get_batches(x,y, batch_size):
    batches = []
    for i in range(0, x.shape[0]/batch_size):
        batches.append((x[i*batch_size:(i+1)*batch_size,:,:,:], y[i*batch_size:(i+1)*batch_size]))
    return batches


#Input data
def load_data(npz='cifar_caffe.npz'):
    f = np.load(npz)
    x_train = f['x_train']
    y_train = f['y_train']
    x_test  = f['x_test']
    y_test  = f['y_test']

    return (x_train, y_train), (x_test, y_test)

def test(net, input_node, output_node, X, y, batch_size, train_or_test):
    batches = get_batches(X, y, batch_size)
    correct_test = 0
    cum_t = 0.0
    total_time = 0
    fmt_str = 'Evaluation [%s]. Batch %d/%d (%d%%). Speed = %.2f sec/b, %.2f img/sec. Batch_precision = %.2f'
    for i, (data, label) in enumerate(batches):
        input_data = data
        net.blobs['data'].data[...] = input_data
        t0 = time.time()
        net.forward(start=input_node)
        t1 = time.time()
        total_time = total_time + (t1 - t0)
        duration = t1 - t0
        cum_t += duration
        sec_per_batch = duration
        img_per_sec   = batch_size/duration
        output = net.blobs[output_node].data
        correct_percent = ((np.argmax(output, axis=1)==np.argmax(label, axis=1)).sum())*100/batch_size
        correct_test += correct_percent
        if cum_t > 0.5:
            sys.stdout.write('\r' + fmt_str %(
                train_or_test, 
                i + 1, 
                len(batches),
                int((i+1)*100/len(batches)),
                sec_per_batch,
                img_per_sec,
                correct_percent
            ))
            sys.stdout.flush()
            cum_t = 0.0
    sys.stdout.write('\r' + fmt_str %(
        train_or_test, 
        i + 1, 
        len(batches),
        int((i+1)*100/len(batches)),
        sec_per_batch,
        img_per_sec,
        correct_percent
    ))
    sys.stdout.write('\n\n%s  Precision = %.2f.\n' % (
    train_or_test,
    correct_test/float(len(batches))
    ))
    print("Average Frames Per Second : %.2f over %d images\n\n" % (((i+1)*batch_size)/(total_time), (i+1)*batch_size))


def main(argv):
    model_def = 'caffe_emdnn.prototxt'
    model_weights = 'caffe_emdnn.caffemodel'
    npz_data = 'cifar_caffe.npz'
    batch_size = 100

    try :
        opts, args = getopt.getopt(argv, "h", ["model_def=", "model_weights=", "npz_data=", "batch_size="])
    except getopt.GetoptError:
        print('Use as : python eval_model.py --model_def=<model.prototxt> --model_weights=<model.caffemodel> --npz_data=<data.npz> --batch_size=<batch_size>')
        sys.exit(2)
    for opt, arg in opts :
        if opt == '-h':
            print('Use as : python eval_model.py --model_def=<model.prototxt> --model_weights=<model.caffemodel> --npz_data=<data.npz> --batch_size=<batch_size>')
            sys.exit()
        elif opt in ("--model_def"):
            model_def = arg
        elif opt in ("--model_weights"):
            model_weights = arg
        elif opt in ("--npz_data"):
            npz_data = arg
        elif opt in ("--batch_size"):
            batch_size = int(arg) 

    print("==========================================>")
    print("Loading model     : "+ model_def)
    print("Loading weights   : "+ model_weights)
    print("Loading data from : "+ npz_data)
    print("Batch Size        : "+ str(batch_size))
    # Create network object
    net = caffe.Net(model_def, model_weights, caffe.TEST)
    print("Total Parameters  : %.3fK" % (sum([np.prod(v[0].data.shape) for k,v in net.params.items()])/1000.0))
    # Get input and output nodes from network
    input_node  = net.top_names.values()[1][0]
    output_node = net.top_names.values()[-1][0]
    # Load npz data (Data is normalized by 255 and mean subtracted in NCHW format)
    (x_train, y_train), (x_test, y_test) = load_data(npz=npz_data)

    # print("==========================================>")
    # print("Evaluating on Training Data")
    # test(net, input_node, output_node, x_train, y_train, batch_size, 'TRAIN DATA')

    print("==========================================>")
    print("Evaluating on Testing Data")
    test(net, input_node, output_node, x_test, y_test, batch_size, 'TEST DATA')

if __name__ == "__main__":
    main(sys.argv[1:])
