import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
from skimage import util
from skimage import measure
import cv2
from PIL import Image
from keras.utils import Sequence
import matplotlib.pyplot as plt
import sys
import shutil
import png
import itertools
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import tqdm 
import imgaug
import pypng
import pillow

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)
    

def crop_or_pad_slice_to_size(slice, nx, ny):

    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def makefolder(folder):
    '''
    Helper function to make a new folder if doesn't exist
    :param folder: path to new folder
    :return: True if folder created, False if folder already exists
    '''
    if not os.path.exists(folder):
        os.makedirs(folder)
        return True
    return False
    
    
def prepare_data(input_folder, output_file, size):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''
    hdf5_file = h5py.File(output_file, "w")
    nx, ny = size
    trainA_addrs = []
    trainB_addrs = []
    
    trainA_path = os.path.join(input_folder, 'trainA')
    trainB_path = os.path.join(input_folder, 'trainB')
    
    pngA_path = os.path.join(trainA_path, 'png')
    pngB_path = os.path.join(trainB_path, 'png')
    
    for pazA, pazB in zip(sorted(os.listdir(pngA_path)), sorted(os.listdir(pngB_path))):
        
        pazA_path = os.path.join(pngA_path, pazA)
        pazB_path = os.path.join(pngB_path, pazB)
              
        pathA = os.path.join(pazA_path, '*.png')
        pathB = os.path.join(pazB_path, '*.png')
        
        for img in sorted(glob.glob(pathA)):
            trainA_addrs.append(img)
        for img in sorted(glob.glob(pathB)):
            trainB_addrs.append(img)
    
    logging.info('Preparing hdf5_file...')
    
    trainA_shape = (len(trainA_addrs), nx, ny)
    trainB_shape = (len(trainB_addrs), nx, ny)
    
    hdf5_file.create_dataset("trainA", trainA_shape, dtype=np.float32)
    hdf5_file.create_dataset("trainB", trainB_shape, dtype=np.float32)
    
    for i in range(len(trainA_addrs)):
        addr_img = trainA_addrs[i]
        img = cv2.imread(addrs_img, 0)
        img = standardize_image(img)
        img = crop_or_pad_slice_to_size(img, 256, 256)
        img = cv2.resize(img, (nx,ny), interpolation=cv2.INTER_AREA)
        hdf5_file["trainA"][i, ...] = img[None]
    
    for i in range(len(trainB_addrs)):
        addr_img = trainB_addrs[i]
        img = cv2.imread(addrs_img, 0)
        img = standardize_image(img)
        img = crop_or_pad_slice_to_size(img, 256, 256)
        img = cv2.resize(img, (nx,ny), interpolation=cv2.INTER_AREA)
        hdf5_file["trainB"][i, ...] = img[None]    
    
    hdf5_file.close()
    
    
def load_data (input_folder,
               preprocessing_folder,
               size,
               force_overwrite=True):
    
    logging.info('input folder:')
    logging.info(input_folder)
    logging.info('output folder:')
    logging.info(preprocessing_folder)
    
    size_str = '_'.join([str(i) for i in size])
    file_name = 'data_2D_size_%s.hdf5' % (size_str)
    file_path = os.path.join(preprocessing_folder, file_name)
    
    makefolder(preprocessing_folder)
    
    if not os.path.exists(file_path) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, file_path, size)
    
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(file_path, 'r')


 if __name__ == '__main__':
    
    # Paths settings
    input_folder = '/content/drive/My Drive/PACEmaker/data'
    preprocessing_folder = '/content/drive/My Drive/PACEmaker/preproc_data'
    
    # Make sure that slice size is multiple 4
    image_size = (176, 176)
    
    d=load_data(input_folder, preprocessing_folder, image_size)
