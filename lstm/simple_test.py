from __future__ import print_function
import sys
sys.path.append('../../caffe/python')
import caffe
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', required=True)
    parser.add_argument('--train')

    args = parser.parse_args()
    caffe.set_mode_gpu()
    net = caffe.Net('models/train.prototxt', 1)
    net.copy_from('snapshots_ZF_iter_55000.caffemodel.h5')
    
