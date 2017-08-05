from __future__ import print_function
import yaml
import caffe
import numpy as np
import re

DEBUG = True


class Transform:
    def __init__(self):
        self.means = [103.939, 116.779, 128.68]
        self.shape = (185, 613, 3)
        self.scale = 255.
        
    def process(self, image):
        image = caffe.io.resize_image(image, self.shape)
        image = image.astype(np.float32, copy=False)
        image *= self.scale
        image -= self.means
        image = np.transpose(image, (2, 0, 1))
        return image


class data_layer(caffe.Layer):
    def setup(self, bottom, top):
        self.image_mean = [103.939, 116.779, 128.68]
        self.batch_ix = 0
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

        self.timesteps = int(params['timesteps'])
        self.streams = int(params['streams'])
        width = int(params.get('width', 1226))
        height = int(params.get('height', 370))
        channels = int(params.get('channels', 3))
        self.shape = (self.timesteps*self.streams, 2*channels, height, width)
        top[0].reshape(*self.shape)
        top[1].reshape(self.timesteps*self.streams)
        top[2].reshape(self.timesteps, self.streams)

        self.tf = Transform()
        self.start = 0
                                       
    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        data = []
        delta = np.ones((self.timesteps, self.streams), np.float)
        delta[...] = 1.
        for seq in range(self.streams):
            if self.start+self.timesteps > len(self.frames):
                self.start = 0
            batch_frames = self.frames[self.start:self.start+self.timesteps]
            data.append(batch_frames)
            batch_seqs = [re.search('sequences\/(\d{2})\/', path1).groups(1) for (path1, _, _) in batch_frames]
            delta[1:, seq] = map(lambda x: float(x[0] == x[1]), zip(batch_seqs[1:], batch_seqs[:-1]))
            self.start += self.timesteps
        data = zip(*data)
        image_data, labels = [], []
        if DEBUG:
            print('current batch data:', data[0])
        for j, batch in enumerate(data):
            for data_point in batch:
                image1 = caffe.io.load_image(data_point[0])
                image1 = self.tf.process(image1)
                image2 = caffe.io.load_image(data_point[1])
                image2 = self.tf.process(image2)
                labels.append(data_point[2])
                image_data.append(np.concatenate((image1, image2)))
        image_data = np.array(image_data)

        top[0].data[...] = image_data
        top[1].data[...] = labels
        top[2].data[...] = delta
          
        def backward(self, top, propagate_down, bottom):
            pass
