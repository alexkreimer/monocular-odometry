import argparse
import pandas as pd
import os
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--split', type=float, default=.8)
    parser.add_argument('input_file')

    args = parser.parse_args()
    df = pd.read_csv(args.input_file, header=None, delim_whitespace=True)
    file_path, file_ext = os.path.splitext(args.input_file)

    if args.shuffle:
        train_inds = np.random.rand(len(df)) < args.split
    else:
        split = int(len(df)*args.split)
        train_inds = [True]*split + [False]*(len(df) - split)

    df[train_inds].to_csv('%s_train%s' % (file_path, file_ext), index=False,
                          header=False, sep=' ')
    df[~np.array(train_inds)].to_csv('%s_test%s' % (file_path, file_ext),
                                     index=False, header=False, sep=' ')
    
    
