import caffe
import numpy as np
import matplotlib.pyplot as plt
import argparse


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size
       approx. sqrt(n) by sqrt(n)  """

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    # pad with ones (white)
    data = np.pad(data, padding, mode='constant', constant_values=1)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:])\
               .transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n*data.shape[1], n * data.shape[3]) + data.shape[4:])

    data = data.reshape((36, 36))
    plt.imshow(data)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--state', required=True)
    args = parser.parse_args()
    
    solver = caffe.get_solver(args.solver)
    solver.net.copy_from(args.model)
    solver.restore(args.state)

    # the parameters are a list of [weights, biases]
    filters = solver.net.params['conv1_1'][0].data
    vis_square(filters.transpose(0, 2, 3, 1))
