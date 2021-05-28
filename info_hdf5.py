"""
Created on Tue Jan 26 12:11:38 2021

@author: Marco Penso
"""

import os
import numpy as np
import h5py

path = r'F:\Final_data'

for fold in sorted(os.listdir(path)):
    fold = os.path.join(path, fold)
    
    data_file_path = os.path.join(fold, 'train.hdf5')
    print('file %s' % data_file_path)
    data = h5py.File(data_file_path, 'r')

    print("Keys: %s" % data.keys())

    for i in range(len(data.keys())):
        print(data[list(data.keys())[i]])
        
    data.close()
