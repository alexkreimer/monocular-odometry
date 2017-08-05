import numpy as np
import sys
import matplotlib.pyplot as plt
import kalman
import argparse


def kf(measurements):
    kf = kalman.KalmanFilter()
    x = np.array([[0.], [0.]])  # initial state (location and velocity)
    P = np.array([[1000., 0.], [0., 1000.]])  # initial uncertainty
    posterior = [x]
    xabs = 0
    for measurement in measurements:
        xabs = xabs + measurement
        x, P = kf.step(x, P, xabs)
        posterior.append(x)
    return posterior


def eval_estimation(data, data_gt):
    delta = data - data_gt
    ind = range(len(delta))

    fig, ax = plt.subplots(2, sharex=True)
    p1 = ax[0].plot(data)
    p2 = ax[0].plot(data_gt)
    fig.text(0.5, 0.04, 'frame number ', ha='center', va='center')
    fig.text(0.06, 0.5, 'translation scale', ha='center', va='center',
             rotation='vertical')

    ax[0].legend((p1[0], p2[0]), ('prediction', 'ground truth'))

    p1 = ax[1].bar(ind, data_gt)
    p2 = ax[1].bar(ind, abs(delta), bottom=data_gt)
    ax[1].legend((p1[0], p2[0]), ('ground truth', 'prediction error'))
    fig.savefig('bars_10.eps', format='eps', dpi=1200)

    bias = np.mean(delta)
    unbiased_delta = np.divide(abs(delta-bias), data_gt)
    print 'absolute error stats: delta.mean=', np.mean(delta), ' std=',\
        np.std(delta)
    plt.figure()
    delta = np.divide(delta, data_gt)
    plt.semilogy(100*abs(delta))
    plt.xlabel('frame number')
    plt.ylabel('relative scale error [%]')
    # plt.title('relative (abs) scale error for sequence 10 on a log-scale')
    plt.savefig('relative_scale_error10.eps', format='eps', dpi=1200)

    plt.figure()
    plt.hist(100*delta, bins=100)
    plt.xlabel('relative error [%]')
    # plt.title('relative error (unnormalized) distribution for sequence 10')
    plt.savefig('scale_error_distrib10.eps', format='eps', dpi=1200)
    print 'mean error: ', np.mean(abs(delta)), 'unbiased mean error: ',\
        np.mean(unbiased_delta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', required=True)
    parser.add_argument('--list_file', required=True)
    args = parser.parse_args()
    
    with open(args.prediction) as fd:
        data = np.array([float(d) for d in fd.read().split()])
    with open(args.list_file) as fd:
        data_gt = np.array([float(d) for d in fd.read().split()[2::3]])

    kf_data = kf(data)
    kf_data = np.array([val[0] for val in kf_data]).ravel()
    kf_data = kf_data[1:] - kf_data[:-1]

    eval_estimation(kf_data, data_gt)
    plt.show()
