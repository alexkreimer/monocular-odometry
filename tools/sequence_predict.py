#!/usr/bin/env python
from __future__ import print_function
import argparse
import sys
import caffe


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', required=True)
    parser.add_argument('--frame_num', required=True, type=int)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    
    args = parser.parse_args()
    
    caffe.init_log()
    caffe.set_device(0)
    caffe.set_mode_gpu()

    net = caffe.Net(args.proto, caffe.TEST)
    net.copy_from(args.model)
    
    print('blobs:', file=sys.stderr)
    for layer_name, blob in net.blobs.iteritems():
        print(layer_name + '\t' + str(blob.data.shape), file=sys.stderr)
    print('params:', file=sys.stderr)
    for layer_name, param in net.params.iteritems():
        print(layer_name + '\t' + str(param[0].data.shape),
              str(param[1].data.shape), file=sys.stderr)
    scales = []
    for i in range(args.frame_num):
        net.forward()
        scales.append(float(net.blobs['fc8'].data))
    with open(args.output, 'w+') as fd:
        fd.write('\n'.join([str(scale) for scale in scales]))
