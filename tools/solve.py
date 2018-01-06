import os
#os.environ['GLOG_minloglevel'] = '2'
from tqdm import trange
import caffe
import argparse
import pandas as pd


def main(solver_proto, out_dir):
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver(solver_proto)
    train_loss, test_loss = [], []
    test_loss.append(solver.test_nets[0].blobs['loss'].data.copy())
    for ix in trange(solver.param.max_iter, desc='overall progress'): #):
        for jx in trange(solver.param.test_interval, desc='until next test'):
            solver.step(1)
            train_loss.append(solver.net.blobs['loss'].data.ravel()[0])
        test_loss.append(solver.test_nets[0].blobs['loss'].data.ravel()[0])
        if ix % 1 == 0:
            solver.snapshot()
            pd.DataFrame(train_loss, columns=['train_loss']).to_csv(os.path.join(out_dir, 'train_loss.csv'))
            pd.DataFrame(test_loss, columns=['test_loss']).to_csv(os.path.join(out_dir, 'test_loss.csv'))                                                                   


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', required=True, type=str)
    parser.add_argument('--out_dir', default='.')
    args = parser.parse_args()

    main(args.solver, args.out_dir)
