#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys
import caffe
import lmdb
import progressbar
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    caffe.init_log()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(network_file=args.proto,
                    phase=caffe.TEST,
                    level=0, weights=args.weights, s tages=['test_on_train'])
    env = lmdb.open('workspace/lmdb/labels_00_shuf_train')
    num_examples = int(env.stat()['entries'])
    env.close()

    print('{} examples in lmdb'.format(num_examples))
    
    #print('blobs:', file=sys.stderr)
    #for layer_name, blob in net.blobs.iteritems():
    #    print(layer_name + '\t' + str(blob.data.shape), file=sys.stderr)
    #print('params:', file=sys.stderr)
    #for layer_name, param in net.params.iteritems():
    #    print(layer_name + '\t' + str(param[0].data.shape),
    #          str(param[1].data.shape), file=sys.stderr)
    scales, gt_scales = [], []
    batch_size = 5
    num_iter = num_examples/batch_size
    with progressbar.ProgressBar(max_value=num_iter) as progress:
        for i in range(num_iter):
            net.forward()
            import ipdb; ipdb.set_trace()
            scales.append(net.blobs['scale'].data.copy().ravel())
            gt_scales.append(net.blobs['label'].data.copy().ravel())
            progress.update(i)
    pred = np.array(scales).ravel()
    gt = np.array(gt_scales).ravel()

    
