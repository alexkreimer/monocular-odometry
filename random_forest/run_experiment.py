from __future__ import print_function
import features_from_list
import fit
import os
import yaml
import argparse
import multiprocessing as mp
import gzip
import pickle
import numpy as np
from tools.misc import make_sure_path_exists, sequence_from_path
import math
import pandas as pd
from collections import namedtuple
import xgboost as xgb

def load_config(experiment_file):
    with open(experiment_file, 'r') as fd:
        config = yaml.load(fd)
    if 'name' not in config:
        config['name'] = os.path.splitext(
            os.path.basename(args.experiment_file))[0]
    if 'rf_size' not in config:
        config['rf_size'] = 10

    for field in ['grid_x', 'grid_y', 'bins']:
        if field in config:
            try:
                if ':' in config[field]:
                    begin, end, step = [int(val) for val in config[field].split(':')]
                    config[field] = list(range(begin, end, step))
            except TypeError:
                config[field] = [config[field]]
    return config
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', required=True,
                        help='relative to the workspace')
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()

    config = load_config(args.experiment_file)

    print(config)
    datasets = features_from_list.main(config, args.workspace)
    print(datasets)
    workers = mp.cpu_count()

    print('fiting models...')
    Model = namedtuple('Model', 'tag, instance, needs_meta, rescale')
    models = [Model('ETR', fit.ExtraTreesRegressor2(n_jobs=workers, n_estimators=config['rf_size']), True, False),
              Model('XGB', xgb.XGBRegressor(n_jobs=mp.cpu_count()*2, max_depth=100, n_estimators=10000, silent=False, nrounds=10), False, False)
              # Model('ETR_scaled', fit.ExtraTreesRegressor2(n_jobs=workers), True, True)
    ]
    output_dir = os.path.join(args.workspace, 'output', config['name'], 'model')
    make_sure_path_exists(output_dir)
    result_dir = os.path.join(args.workspace, 'output', config['name'], 'predict')
    make_sure_path_exists(result_dir)

    for key, dataset in datasets.items():
        print('reading', dataset['train_list'])
        data = pickle.load(gzip.open(dataset['train_list'], 'rb'))
        X, meta = data['features'], data['meta']
        y = np.array([float(m[2]) for m in meta])
        pairs = [m[:2] for m in meta]
        print('reading', dataset['test_list'])
        data_test = pickle.load(gzip.open(dataset['test_list'], 'rb'))
        X_test, meta_test = data_test['features'], data_test['meta']
        y_test = np.array([float(m[2]) for m in meta_test])
        pairs_test = [m[:2] for m in meta_test]
        for model in models:
            sha = '%s_%d_%d_%d' % (model[0], key[0], key[1], key[2])
            model_binary = os.path.join(output_dir, '%s.pklz' % sha)
            scaler_binary = os.path.join(output_dir, 'scaler_%s.pklz' % sha)

            if not os.path.isfile(model_binary) or not os.path.isfile(scaler_binary):
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
                print('read model binary %s...' % model_binary)
                print('read scaler binary %s...' % scaler_binary)

            # compute the predictions the validation data
            y_pred = fit.predict(clf, scaler, X_test, pairs_test, model)
            assert(len(y_pred) == len(y_test))
            # errors = np.abs(y_pred - y_test)
            # sorted_errors = np.argsort(errors)[::-1]
            # sequences = []
            # for idx in sorted_errors:
            #    sequences.append(sequence_from_path(pairs_test[idx][0]))
            # lf = []
            # for idx in sorted_errors:
            #     leaf_nodes = clf.leaf_items([X_test[idx]])
            #     lf.append([ln.pairs for ln in leaf_nodes[0]])

            df = pd.DataFrame({'y': y_test, 'y_pred': y_pred})
            # 'y_error': np.abs(y_pred[sorted_errors] - y_test[sorted_errors]),
            # 'pairs': [pairs_test[idx] for idx in sorted_errors]})
            # leaf_pairs': lf})
            df.to_csv(os.path.join(result_dir, '%s_test.txt' % sha))

            train_pred = fit.predict(clf, scaler, X, pairs, model)
            df = pd.DataFrame({'y': y, 'y_pred': train_pred})
            df.to_csv(os.path.join(result_dir, '%s_train.txt' % sha))
