import pickle
import tables
import h5py

txt_file = 'data/test/test.apiseq.txt'
h5_file = 'data/test/test.apiseq.h5'

with open(txt_file, 'r') as f, h5py.File(h5_file, 'w+') as hfw:
    for line in f.readlines():
        hfw['phrase'] = line
