from __future__ import print_function
import features_from_list
import fit
import os
import yaml
import argparse
import multiprocessing as mp
from sklearn.ensemble import ExtraTreesRegressor
import gzip
import pickle
import numpy as np
from tools.misc import make_sure_path_exists, sequence_from_path
import math
import pandas as pd
from collections import namedtuple


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', required=True,
                        help='relative to the workspace')
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()

    with open(args.experiment_file, 'r') as fd:
        config = yaml.load(fd)
    if 'name' not in config:
        config['name'] = os.path.splitext(
            os.path.basename(args.experiment_file))[0]
    if 'rf_size' not in config:
        config['rf_size'] = 10

    print(config)
    dataset = features_from_list.main(config, args.workspace)
    print(dataset)
    workers = mp.cpu_count()

    Model = namedtuple('Model', 'tag, instance, needs_meta, rescale')
    models = [Model('ETR',
                    fit.ExtraTreesRegressor2(n_jobs=workers,
                                             n_estimators=config['rf_size']),
                    True, False),
              Model('ETR_scaled', fit.ExtraTreesRegressor2(n_jobs=workers),
                    True, True)]
    data = pickle.load(gzip.open(dataset['train_list'], 'rb'))
    X, meta = data['features'], data['meta']
    y = np.array([float(m[2]) for m in meta])
    pairs = [m[:2] for m in meta]
    data_test = pickle.load(gzip.open(dataset['test_list'], 'rb'))
    X_test, meta_test = data_test['features'], data_test['meta']
    y_test = np.array([float(m[2]) for m in meta_test])
    pairs_test = [m[:2] for m in meta_test]
    output_dir = os.path.join(args.workspace, 'output', config['name'],
                              'model')
    make_sure_path_exists(output_dir)
    result_dir = os.path.join(args.workspace, 'output', config['name'],
                              'predict')
    make_sure_path_exists(result_dir)
    for model in models:
        model_binary = os.path.join(output_dir, '%s.pklz' % model[0])
        scaler_binary = os.path.join(output_dir, 'scaler_%s.pklz' % model[0])

        if not os.path.isfile(model_binary) or\
           not os.path.isfile(scaler_binary):
            scaler, clf = fit.fit_model(X, y, pairs, model)

            with gzip.open(scaler_binary, 'wb') as fd:
                pickle.dump({'scaler': scaler}, fd)

            with gzip.open(model_binary, 'wb') as fid:
                pickle.dump({'clf': clf}, fid)

            print('wrote %s...' % model_binary)
            print('scaler %s...' % scaler_binary)
        else:
            with gzip.open(scaler_binary, 'rb') as fd:
                data = pickle.load(fd)
                scaler = data['scaler']

            with gzip.open(model_binary, 'rb') as fd:
                data = pickle.load(fd)
                clf = data['clf']
            
        # compute the predictions the validation data
        y_pred = fit.predict(clf, scaler, X_test, pairs_test, model)
        assert(len(y_pred) == len(y_test))

        errors = np.abs(y_pred - y_test)
        sorted_errors = np.argsort(errors)[::-1]
        sequences = []
        for idx in sorted_errors:
            sequences.append(sequence_from_path(pairs_test[idx][0]))

        lf = []
        for idx in sorted_errors:
            leaf_nodes = clf.leaf_items([X_test[idx]])
            lf.append([ln.pairs for ln in leaf_nodes[0]])

        df = pd.DataFrame({'y': y_test[sorted_errors],
                           'y_pred': y_pred[sorted_errors],
                           'y_error': np.abs(y_pred[sorted_errors] - y_test[sorted_errors]),
                           'pairs': [pairs_test[idx] for idx in sorted_errors],
                           'leaf_pairs': lf})
        
        df.to_csv(os.path.join(result_dir, '%s_errors_full.txt' % model[0]))
        
        df = pd.DataFrame({'y': y_test, 'y_pred': y_pred})
        df.to_csv(os.path.join(result_dir, '%s_errors.txt' % model[0]))
        
        errors = np.linalg.norm(y_test-y_pred)
        rms = math.sqrt(errors*errors/len(y))
        print(model[0], 'rms:', rms)
