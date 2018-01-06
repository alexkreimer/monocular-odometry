from __future__ import print_function
import os
import sys
import errno
import warnings
import traceback
import pickle
import gzip
import cv2
import multiprocessing as mp
import numpy as np
# import matplotlib.pyplot as plt
import yaml
from collections import namedtuple
from PIL import Image
from progressbar import Percentage, Bar, ProgressBar
import argparse
import pandas as pd
from tools.misc import make_sure_path_exists
import itertools

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

KITTI_HOME = '/data/other/'

Intrinsic = namedtuple('Intrinsic', ['f', 'pp'])


def sequence_from_path(path):
    return os.path.normpath(path).split(os.sep)[5]


def read_image(path):
    pil_image = Image.open(path).convert('L')
    pil_image = pil_image.resize((1226, 370), Image.ANTIALIAS)
    return np.array(pil_image)


def patch_extractor(a, img, win_size):
    ''' extract a patch from the image '''
    row_min, row_max = a[1]-win_size, a[1]+win_size
    col_min, col_max = a[0]-win_size, a[0]+win_size
    val = img[np.ix_(range(row_min, row_max+1), range(col_min, col_max+1))]
    return val.ravel()


def harris(gray):
    # image size
    rows, cols = gray.shape

    # gray is currently 'uint8', convert it to 'float32'
    gray = np.float32(gray)

    # apply the detector
    kp = cv2.goodFeaturesToTrack(gray, maxCorners=2000, qualityLevel=.01,
                                 minDistance=3)
    kp = np.reshape(kp, (kp.shape[0], 2))

    # strip features too close to the edges, since we need to extract
    # patches
    thresh = 5
    good_x = np.logical_and(kp[:, 0] > thresh, kp[:, 0] < cols-thresh)
    good_y = np.logical_and(kp[:, 1] > thresh, kp[:, 1] < rows-thresh)

    # compute the keypoints
    kp = kp[np.logical_and(good_x, good_y)].astype('uint32')

    # compute the descriptors
    d = np.apply_along_axis(lambda x: patch_extractor(x, gray, thresh), 1, kp)
    return (kp, d)


def match(des1, des2):
    # brute force L1 matcher
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def prune_matches(matches, kp1, kp2, des1, des2, intr):
    matches = sorted(matches, key=lambda x: x.distance)
    
    m1 = np.array([m.queryIdx for m in matches])
    m2 = np.array([m.trainIdx for m in matches])
    
    # slice inliers
    kp1_matched = kp1[m1, :]
    kp2_matched = kp2[m2, :]

    des1_matched = des1[m1, :]
    des2_matched = des2[m2, :]

    E, inliers = cv2.findEssentialMat(kp1_matched.astype('float32'),
                                      kp2_matched.astype('float32'),
                                      intr.f, tuple(intr.pp))
    # retval, R, t, mask = cv2.recoverPose(E, kp1, kp2)
    inliers = inliers.ravel().view('bool')
    kp1_matched = kp1_matched[inliers, :]
    kp2_matched = kp2_matched[inliers, :]
    des1_matched = des1_matched[inliers, :]
    des2_matched = des2_matched[inliers, :]
    return (kp1_matched, des1_matched, kp2_matched, des2_matched)


def compute_hist(corners, nbins):
    '''uniform bins 0-200 '''
    deltas = corners[0].astype('int32') - corners[1].astype('int32')
    deltas = np.linalg.norm(deltas, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        h, edges = np.histogram(deltas, bins=range(nbins+1), density=False)
    h = np.array(np.nan_to_num(h))
    return (h, edges)


def compute_feature_vector(kp1, kp2, grid, nbins, pair):
    image1 = read_image(pair[0])
    image2 = read_image(pair[1])
    assert (image1.shape == image2.shape),\
        'input images are of different shape'

    assert len(image1.shape) == 2 or image1.shape[2] == 1

    # edges contain boundaries, i.e., the number of cells will be #-2
    xedges = np.linspace(0, image1.shape[1], num=grid[0]+1)
    yedges = np.linspace(0, image1.shape[0], num=grid[1]+1)

    xv, yv = np.meshgrid(xedges, yedges, indexing='ij')

    # seq = sequence_from_path(pair[0])
    # basename1 = os.path.splitext(os.path.basename(pair[0]))[0]
    # basename2 = os.path.splitext(os.path.basename(pair[1]))[0]
    # basename = '_'.join([seq, basename1, basename2])

    # fig, ax = plt.subplots(grid[1], grid[0], sharex=True, sharey=True)
    desc = []
    for i in range(grid[0]):
        for j in range(grid[1]):
            xmin, xmax = xv[i, j], xv[i+1, j]
            ymin, ymax = yv[i, j], yv[i, j+1]

            xbin = np.logical_and(kp1[:, 0] > xmin, kp1[:, 0] <= xmax)
            ybin = np.logical_and(kp1[:, 1] > ymin, kp1[:, 1] <= ymax)
            val = np.logical_and(xbin, ybin)
            # binned feature points
            kp1_bin = kp1[val, :]
            kp2_bin = kp2[val, :]
            (h, edges) = compute_hist((kp1_bin, kp2_bin), nbins=nbins)
            # ax[j, i].bar(xrange(h.shape[0]), h, align='center')
            # ax[j, i].plot(h)
            desc.append(h)

    # fig.suptitle('bins of a feature vector')
    # fig.savefig(os.path.join('data', 'debug',
    #                          '%s_bins_feature_vector.png' % basename))
    # plt.close(fig)
    desc = np.array(desc).ravel()
    
    # fig, (ax1, ax2) = plt.subplots(2, 1)
    # blend = image1*.5 + image2*.5
    # ax1.imshow(blend, cmap='gray')
    # ax1.scatter(kp1[:, 0], kp1[:, 1], s=3, c='red')
    # ax1.scatter(kp2[:, 0], kp2[:, 1], s=3, c='green')
    # ax1.plot([kp1[:, 0], kp2[:, 0]], [kp1[:, 1], kp2[:, 1]])

    # for x in xedges:
    #     ax1.plot([x, x], [0, yedges[-1]])
    # for y in yedges:
    #     ax1.plot([0, xedges[-1]], [y, y])
    
    # ax1.axis('off')
    # fig.suptitle('feature vector grid')
    
    # ax2.plot(desc)
    # # ax2.bar(xrange(desc.shape[0]), desc)
    # ax2.set_title('complete feature vector')
    # fig.savefig(os.path.join('data', 'debug',
    #                          '%s_feature_vector.png' % basename))
    return desc


def warn_with_traceback(message, category, filename, lineno, file=None,
                        line=None):
    traceback.print_stack()
    log = file if hasattr(file, 'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno,
                                     line))


# warnings.showwarning = warn_with_traceback


def read_intrinsics(seq):
    ''' read KITTI camera intrinsics for a sequence '''
    global KITTI_HOME

    calib_file = os.path.join(KITTI_HOME, 'dataset', 'sequences', seq,
                              'calib.txt')
    with open(calib_file, 'r') as f:
        for line in f:
            line = [float(v) for v in line.split()[1:]]
            K = np.array(line).reshape(3, 4)
            break
    intr = Intrinsic(f=K[0, 0], pp=K[range(2), 2])
    return intr


def extract_match_corners_pair(pair):
    global intrinsics
    ''' compute harris corners and match them '''
    try:
        image1 = read_image(pair[0])
        image2 = read_image(pair[1])
        assert (image1.shape == image2.shape),\
            'input images are of different shape'

        assert len(image1.shape) == 2 or image1.shape[2] == 1
        seq = sequence_from_path(pair[0])
        # basename1 = os.path.splitext(os.path.basename(pair[0]))[0]
        # basename2 = os.path.splitext(os.path.basename(pair[1]))[0]
        # basename = '_'.join([seq, basename1, basename2])

        (kp1, des1) = harris(image1)
        (kp2, des2) = harris(image2)

        # fig, (ax1, ax2) = plt.subplots(2, 1)
        # for image, axis, keypoints in [(image1, ax1, kp1), (image2, ax2, kp2)]:
        #     axis.imshow(image, cmap='gray')
        #     axis.scatter(keypoints[:, 0], keypoints[:, 1], s=3)
        #     axis.axis('off')
        # fig.suptitle('raw corners')
        matches = match(des1, des2)
        kp1, des1, kp2, des2 = prune_matches(matches, kp1, kp2, des1, des2,
                                             read_intrinsics(seq))
        # blend = image1*.5 + image2*.5
        # fig.savefig(os.path.join('data', 'debug',
        #                          '%s_raw_corners.png' % basename))
        # plt.close(fig)
        # fig, ax = plt.subplots()
        # ax.imshow(blend, cmap='gray')
        # ax.scatter(kp1[:, 0], kp1[:, 1], s=3, c='red')
        # ax.scatter(kp2[:, 0], kp2[:, 1], s=3, c='green')
        # ax.plot([kp1[:, 0], kp2[:, 0]], [kp1[:, 1], kp2[:, 1]])
        # ax.axis('off')
        # fig.suptitle('final corner match')
        # fig.savefig(os.path.join('data', 'debug',
        #                          '%s_final_matches.png' % basename))
        # plt.close(fig)
    except:
        print('error: %s' % traceback.format_exc())
        kp1 = None
        kp2 = None
    return (kp1, kp2, pair)


def extract_match_corners_sequence(image_data):
    ''' extract corners and match them between image pairs '''
    bar = ProgressBar(widgets=[Percentage(), Bar()],
                      max_value=len(image_data)).start()
    corners = []
    p = mp.Pool(2*mp.cpu_count())
    async_results = [p.map_async(extract_match_corners_pair, (t,),
                                 callback=corners.extend) for t in image_data]
    p.close()
    while len(corners) < len(image_data):
        bar.update(len(corners))
        # make sure subprocess exceptions are re-raised
        for x in async_results:
            try:
                x.get(timeout=.001)
            except mp.TimeoutError:
                pass
    bar.update(len(corners))
    bar.finish()
    p.join()
    return corners


def compute_feature_vector_wrapper(args):
    ''' this is here because multiprocessing uses pickle which
    does not pickle lambdas '''
    kp1, kp2, pair, grid, nbins = args
    return compute_feature_vector(kp1, kp2, grid, nbins, pair)


def compute_features(corners, grid=(6, 4), nbins=300):
    X = []
    bar = ProgressBar(widgets=[Percentage(), Bar()],
                      max_value=len(corners)).start()
    p = mp.Pool(2*mp.cpu_count())
    tasks = [(corner[0], corner[1], corner[2], grid, nbins)
             for corner in corners]

    for x in tasks:
        compute_feature_vector_wrapper(x)
    async_results = [p.map_async(compute_feature_vector_wrapper, (x,),
                                 callback=X.extend) for x in tasks]
    p.close()
    while len(X) < len(corners):
        bar.update(len(X))
        for x in async_results:
            try:
                x.get(timeout=0.001)
            except mp.TimeoutError:
                pass
    bar.update(len(X))
    bar.finish()
    p.join()
    return X

    
def load_images(pair):
    im1 = read_image(pair[0])
    im2 = read_image(pair[1])
    return (im1, im2)


def main(config, workspace):
    datasets = {}

    output_dir = os.path.join(workspace, 'output', config['name'], 'features')
    make_sure_path_exists(output_dir)    
    for grid in itertools.product(config['grid_x'], config['grid_y']):
        for bins in config['bins']:
            dataset = {}            
            for key in ('train_list', 'test_list'):
                features_file = os.path.join(output_dir, '%s_%d_%d_%d.pklz' % (key, grid[0], grid[1], bins))
                print('features_file', features_file)                
                if os.path.isfile(features_file):
                    print('found features file', features_file)
                    dataset[key] = features_file
                    datasets[(grid[0], grid[1], bins)] = dataset                    
                    continue
                else:
                    print('creating', features_file)
                df = pd.read_csv(os.path.join(workspace, config[key]),
                                 delim_whitespace=True, header=None)
                image_list = df.loc[:, [0, 1]].values.tolist()
                print('extracting corners')
                corners = extract_match_corners_sequence(image_list)
                print('compute features')
                features = compute_features(corners, grid, bins)
                with gzip.open(features_file, 'wb+') as fd:
                    pickle.dump({'features': features, 'meta': df.values.tolist()}, fd)
                dataset[key] = features_file
                datasets[(grid[0], grid[1], bins)] = dataset
    return datasets


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_file', required=True,
                        help='relative to the workspace')
    parser.add_argument('--workspace', required=True)
    args = parser.parse_args()

    with open(args.experiment_file, 'r') as fd:
        config = yaml.load(fd)
    if 'name' not in config:
        config['name'] = os.path.splitext(os.path.basename(args.experiment_file))[0]
    main(config, args.workspace)
