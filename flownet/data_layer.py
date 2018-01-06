from __future__ import print_function
import yaml
import caffe
import numpy as np
import re

DEBUG = False

class Transform:
    def __init__(self, shape):
        self.mean = 103.939/255
        self.shape = shape
        
    def process(self, image):
        image = caffe.io.resize_image(image, self.shape)
        image -= self.mean
        image = np.transpose(image, (2, 0, 1))
        return image


class data_layer(caffe.Layer):
    def setup(self, bottom, top):
        params = yaml.load(self.param_str)
        with open(params['list_file'], 'r') as fd:
            data = filter(None, map(lambda x: x.strip(), fd.readlines()))

        self.frames = []
        for ix, line in enumerate(data):
            image1_path, image2_path, label = line.split()
            self.frames.append((image1_path, image2_path, float(label)))
        if DEBUG:
            print('data_layer.setup:')
            print('read', len(data), 'datapoints from', params['list_file'])
        width = int(params.get('width', 1226))
        height = int(params.get('height', 370))
        channels = int(params.get('channels', 3))
        batch_size = int(params.get('batch_size', 5))
        self.shape = (batch_size, 2*channels, height, width)
        top[0].reshape(*self.shape)
        top[1].reshape(batch_size)
        self.tf = Transform(shape=(height, width, channels))
        self.start = 0
                                       
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        batch_size = self.shape[0]
        data = []
        for ix in range(batch_size):
            if self.start == len(self.frames):
                self.start = 0
            data.append(self.frames[self.start])
            self.start += 1
        image_data, labels = [], []
        if DEBUG:
            print('batch data:', data)
        for image1_file, image2_file, label in data:
            image1 = caffe.io.load_image(image1_file, color=True)
            image1 = self.tf.process(image1)
            image2 = caffe.io.load_image(image2_file, color=True)
            image2 = self.tf.process(image2)
            labels.append(label)
            image_data.append(np.concatenate((image1, image2)))

        top[0].data[...] = np.array(image_data)
        top[1].data[...] = np.array(labels)
          
        def backward(self, top, propagate_down, bottom):
            pass
