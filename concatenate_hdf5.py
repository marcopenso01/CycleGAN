"""
Created on Wed Jan 20 18:13:00 2021

@author: Marco Penso
"""

import os
import numpy as np
import h5py

'file to match'
path1 = r'F:\prova\train.hdf5'
path2 = r'F:\data3\trainB\preprocessing\train.hdf5'
path_out = r'F:\prova2'
data_file_path = os.path.join(path_out, 'train.hdf5')

data1 = h5py.File(path1, 'r')
data2 = h5py.File(path2, 'r')
hdf5_file = h5py.File(data_file_path, "w")

print("Keys: %s" % data1.keys())

for i in range(len(data1.keys())):
    d1 = data1[list(data1.keys())[i]][()]
    d2 = data2[list(data2.keys())[i]][()]
    d = np.concatenate((d1, d2), axis=0)
    if d.dtype == ('O'):
        hdf5_file.create_dataset(list(data1.keys())[i], d.shape, dtype=h5py.special_dtype(vlen=str))
    else:
        hdf5_file.create_dataset(list(data1.keys())[i], d.shape, d.dtype)
    hdf5_file[list(data1.keys())[i]][()] = d
    
    print("key: %s" % list(data1.keys())[i])
    print("data1 shape:", d1.shape)
    print("data2 shape:", d2.shape)
    print("after concatenation:", d.shape)

data1.close()
data2.close()
hdf5_file.close()
