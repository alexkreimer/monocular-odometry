#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import sys
from PIL import Image
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format


def update_lr():
    solver_config = caffe_pb2.SolverParameter()
    with open('flow_solver.prototxt') as f:
        text_format.Merge(str(f.read()), solver_config)
    solver_config.base_lr *= .1
    new_solver_config = text_format.MessageToString(solver_config)
    with open('temp_solver.prototxt', 'w+') as f:
        f.write(new_solver_config)
    solver = caffe.get_solver('temp_solver.prototxt')
    return solver


def vis_square(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) thing in a grid of size
       approx. sqrt(n) by sqrt(n)

    """

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
    plt.imshow(data); plt.axis('off')


def test(solver, test_data, epoch):
    loss = []
    scale = []
    labels = []
    epoch_size = len(test_data)
    for i, elem in enumerate(test_data):
        elem = elem.split()
        label = float(elem[2])
        label = np.array(label)
        label = label[np.newaxis, np.newaxis, np.newaxis, np.newaxis, ...]
        transformed_image1 = read_transform_image(elem[0], transformer)
        transformed_image2 = read_transform_image(elem[1], transformer)
        image = np.concatenate([transformed_image1, transformed_image2],
                               axis=1)
        solver.test_nets[0].blobs['data'].reshape(*image.shape)
        solver.test_nets[0].blobs['data'].data[...] = image
        solver.test_nets[0].blobs['label'].reshape(*label.shape)
        solver.test_nets[0].blobs['label'].data[...] = label
        solver.test_nets[0].forward()
        loss.append(2*float(solver.test_nets[0].blobs['loss'].data))
        scale.append(float(solver.test_nets[0].blobs['predict_scale'].data))
        labels.append(float(solver.test_nets[0].blobs['label'].data))
        if i % 100 == 0:
            print('test: processed %d samples of %d' % (i, epoch_size),
                  file=sys.stderr)
    scales_file = 'scales_%03d.txt' % epoch
    with open(scales_file, 'w+') as fd:
        fd.write('\n'.join([str(s) for s in scale]))
    print('wrote %s' % scales_file, file=sys.stderr)
    mean_loss = np.mean(loss)
    print('test epoch loss: mean: %g, std: %g' % (np.mean(loss), np.std(loss)),
          file=sys.stderr)
    return mean_loss


def create_transformer(shape):
    mu = np.load('mean_image.npz')['arr_0']
    mu = np.array([mu.mean()])
    transformer = caffe.io.Transformer({'data': shape})
    # move image channels to outermost dimension
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.array(mu))
    return transformer


def read_transform_image(file_name, transformer):
    image1 = np.array(Image.open(file_name).resize((1241, 376),
                                                   Image.ANTIALIAS))
    image1 = image1[..., np.newaxis]
    transformed_image = transformer.preprocess('data', image1)
    transformed_image = transformed_image[np.newaxis, ...]
    return transformed_image


if __name__ == '__main__':
    caffe.init_log()
    caffe.set_device(0)
    caffe.set_mode_gpu()
    # net = caffe.Net('kitti_monoscale.prototxt', caffe.TRAIN)

    with open('images_train.list', 'r') as fd:
        train_data = [line for line in fd]

    with open('images_test.list', 'r') as fd:
        test_data = [line for line in fd]

    solver = caffe.get_solver('flow_solver.prototxt')
    solver.net.copy_from('flow_iter_40000.caffemodel')
    solver.restore('flow_iter_40000.solverstate')

    print('blobs:', file=sys.stderr)
    for layer_name, blob in solver.net.blobs.iteritems():
        print(layer_name + '\t' + str(blob.data.shape), file=sys.stderr)
    print('params:', file=sys.stderr)
    for layer_name, param in solver.net.params.iteritems():
        print(layer_name + '\t' + str(param[0].data.shape),
              str(param[1].data.shape), file=sys.stderr)

    transformer = create_transformer((1, 1, 376, 1241))
    test_num = 0
    for epoch in range(100):
        print('-o- train epoch: %d' % epoch, file=sys.stderr)
        train_epoch_size = len(train_data)
        for i, elem in enumerate(train_data):
            elem = elem.split()
            transformed_image1 = read_transform_image(elem[0], transformer)
            transformed_image2 = read_transform_image(elem[1], transformer)
            image = np.concatenate([transformed_image1, transformed_image2],
                                   axis=1)
            label = float(elem[2])
            label = np.array(label)
            label = label[np.newaxis, np.newaxis, np.newaxis, np.newaxis, ...]

            solver.net.blobs['data'].reshape(*image.shape)
            solver.net.blobs['data'].data[...] = image

            solver.net.blobs['label'].reshape(*label.shape)
            solver.net.blobs['label'].data[...] = label
            solver.step(1)
            if i % 500 == 0:
                print('train: processed %d of %d' % (i, train_epoch_size),
                      file=sys.stderr)
            if i % 10000 == 0:
                solver.test_nets[0].share_with(solver.net)
                test(solver, test_data, test_num)
                test_num = test_num + 1
        solver = update_lr()
        iter_num = (epoch+1)*40000
        solver.net.copy_from('flow_iter_%d.caffemodel' % iter_num)
        solver.restore('flow_iter_%d.solverstate' % iter_num)
        
    solver.net.save('flow.caffemodel')
