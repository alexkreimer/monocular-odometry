# coding: utf-8
import errno
import numpy as np
import argparse
import pickle
import os
import math
import multiprocessing as mp
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing

from scipy import stats
import gzip


class LeafNode:
    def __init__(self):
        self.X, self.y, self.pairs = [], [], []
        
    def append(self, X, y, pair):
        self.X.append(X.ravel())
        self.y.append(y)
        self.pairs.append(pair)
    

class ExtraTreesRegressor2(ExtraTreesRegressor):
    
    def __init__(self, n_estimators=10, n_jobs=1):
        super(ExtraTreesRegressor2, self).__init__(n_estimators=n_estimators,
                                                   n_jobs=n_jobs)
    
    def fit_leaf_model(self, X, y, pairs):

        # dictionary for each tree in the ensemble
        self.ensemble_leaf_data = [{}] * self.n_estimators
        
        # for each sample
        for x, y_value, pair in zip(X, y, pairs):
            x = x.reshape(1, -1)
            
            # get a list of leafs that each point lands in
            indices = self.apply(x)[0]
        
            # put them in the dictionary
            for tnum, lind in enumerate(indices):
                lnode = self.ensemble_leaf_data[tnum].get(lind, LeafNode())
                lnode.append(x, y_value, pair)
                self.ensemble_leaf_data[tnum][lind] = lnode

    def leaf_items(self, X):
        result = []
        for x in X:
            x = x.reshape(1, -1)
            leaf_indices = self.apply(x)[0]
            result.append([self.ensemble_leaf_data[tnum].get(lind)
                           for tnum, lind in enumerate(leaf_indices)])
        return result

    def fit(self, X, y, pairs):
        model = super(ExtraTreesRegressor2, self).fit(X, y)
        self.fit_leaf_model(X, y, pairs)
        return model


def predict(clf, scaler, X, pairs, model):
    if model[3]:
        X_scaled = scaler.transform(X)
        return clf.predict(X_scaled)
    else:
        return clf.predict(X)


def fit_model(X, y, pairs, model):
    # compute the mean and the std
    scaler = preprocessing.StandardScaler().fit(X)

    # subtract the mean and divide by the std
    if model.rescale:
        X = scaler.transform(X)
    if model.needs_meta:
        model.instance.fit(X, y, pairs)
    else:
        model.instance.fit(X, y)
    return (scaler, model.instance)


def fit_and_save(train_set, model_name, models, rescale=False):
    # load the training and the validation data
    data = pickle.load(gzip.open(train_set, 'rb'))
    X = data['features']
    meta = data['meta']
    y = np.array([float(m[2]) for m in meta])
    pairs = [m[:2] for m in meta]
    # compute the mean and the std
    scaler = preprocessing.StandardScaler().fit(X)
    dir_name = os.path.join('data', 'models')
    make_sure_path_exists(dir_name)
    with gzip.open(
            os.path.join('data', 'models', 'scaler_%s.pklz' %
                         model_name), 'wb') as fd:
        pickle.dump(scaler, fd)
    # subtract the mean and divide by the std
    if rescale:
        X = scaler.transform(X)
    stats_res = stats.describe(y)
    plt.figure()
    n, bins, patches = plt.hist(y, 50, normed=1, facecolor='green',
                                alpha=0.75)
    plt.grid(True)
    plt.title('Target (train) distribution: $\mu=%0.2g,\sigma=%0.2g$'
              % (stats_res.mean, math.sqrt(stats_res.variance)))
    plt.savefig('y_dist_%s.jpg' % model_name)
    for tag, clf, need_pairs in models:
        model_binary = os.path.join(dir_name, '%s_%s.pklz' % (model_name, tag))
        if os.path.isfile(model_binary):
            print('%s already exists, remove first' % model_binary)
        else:
            start = timer()
            if need_pairs:
                clf.fit(X, y, pairs)
            else:
                clf.fit(X, y)
            end = timer()
            print('model fit wall time:', end-start)
            with gzip.open(model_binary, 'wb') as fid:
                pickle.dump(clf, fid)
            print('wrote %s...' % model_binary)
        # compute the predictions the validation data
        # if need_pairs:
        #     y_pred, pairs = clf.predict(X_valid)
        # else:
        #     y_pred = clf.predict(X_valid)
        # predict_output = '%s_%s.txt' % (model_name, tag)
        # with open(predict_output, 'w') as fd:
        #     fd.write('\n'.join([str(y) for y in y_pred]))
        # print('wrote predictions into', predict_output)


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_set', required=True)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--rescale', action='store_true')
    args = parser.parse_args()

    models = [('LRETR', LinearRegETR(), True),
              ('ETR', ExtraTreesRegressor(), False)]
    models = [('ETR', ExtraTreesRegressor(n_jobs=mp.cpu_count()*2), False)]
    
    fit_and_save(args.train_set, args.model_name, models, args.rescale)
