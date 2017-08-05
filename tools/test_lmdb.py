import caffe
import lmdb
import sys
import matplotlib.pyplot as plt


lmdb_env = lmdb.open(sys.argv[1])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe.proto.caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    label = datum.label
    data = caffe.io.datum_to_array(datum)
    print 'label:', label
    print 'data.shape: ', data.shape

    import ipdb; ipdb.set_trace()
