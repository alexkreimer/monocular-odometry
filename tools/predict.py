import caffe
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import lmdb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--proto', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--lmdb')
    parser.add_argument('--num_entries', help='number of data points in test lmdb')
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    
    caffe.set_mode_gpu()

    if not args.num_entries:
        if args.lmdb:
            env = lmdb.open(args.lmdb)
            num_entries = env.stat()['entries']
    else:
        num_entries = args.num_entries

    net = caffe.Net(args.proto, args.weights, caffe.TEST)
    batch_size = net.blobs['scale'].shape[0]

    pred_scales, gt_scales = [], []
    for i in range(num_entries/batch_size):
        result = net.forward(['scale'])
        pred_scales += result['scale'].ravel().tolist()
        gt_scales += net.blobs['label'].data.ravel().tolist()
    df = pd.DataFrame({'gt_scales': pd.Series(gt_scales), 'pred_scale': pd.Series(pred_scales)})
    df.to_csv(args.output)
