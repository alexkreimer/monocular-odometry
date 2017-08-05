import lmdb
import caffe
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import sys


env = lmdb.open(sys.argv[1])
with env.begin() as txn:
    for key, value in txn.cursor():
        file1 = key[9:60]
        file2 = key[61:]
        image1 = misc.imread(file1)
        image2 = misc.imread(file2)

        f, ax = plt.subplots(2, sharex=True)
        ax[0].imshow(image1)
        ax[1].imshow(image2)

        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(value)
        print 'label:', datum.label
        import ipdb; ipdb.set_trace()
        blob = np.fromstring(datum.data, dtype=np.uint8)
        blob = np.reshape(blob, (datum.channels, datum.height, datum.width), order='C')
        blob = np.transpose(blob, (1, 2, 0))

        image1 = blob[..., :3]
        image2 = blob[..., 3:]
        print image1.shape
        print image2.shape

        f, ax = plt.subplots(2, sharex=True)
        ax[0].imshow(image1)
        ax[1].imshow(image2)
        plt.show()
