import os
import sys
#this file should be run from {caffe_root}/examples (otherwise change this line)
caffe_root = '/home/akreimer/prj/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
# If you get "No module named _caffe", either you have not built pycaffe or you have the wrong path.

import numpy as np
import matplotlib.pyplot as plt


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size approx.
    sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
                (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    
    plt.figure()
    plt.imshow(data)
    plt.axis('off')


if __name__ == '__main__':
    caffe.set_mode_cpu()

    model_def = 'kitti_flow_rgb.prototxt'
    model_weights = 'flow_iter_99000.caffemodel'

    net = caffe.Net(model_def,      # defines the structure of the model
                    model_weights,  # contains the trained weights
                    caffe.TEST)     # use test mode (e.g., don't perform dropout)

    for layer_name, blob in net.blobs.iteritems():
            print layer_name + '\t' + str(blob.data.shape)

    total = 0
    for layer_name, param in net.params.iteritems():
        total += np.prod(param[0].data.shape) + param[1].data.shape
        print layer_name + '\t' + str(param[0].data.shape),\
            str(param[1].data.shape), np.prod(param[0].data.shape)\
            + param[1].data.shape

    print 'total params number: ', total

    filters = net.params['conv1'][0].data
    filters = filters.transpose(0, 2, 3, 1)
    data1 = filters[..., :3]
    data2 = filters[..., 3:]
    vis_square(data1)
    vis_square(data2)
    plt.show()
