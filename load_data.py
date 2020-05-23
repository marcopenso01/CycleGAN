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
import pydicom # for reading dicom files
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)
    

def crop_or_pad_slice_to_size(im, nx, ny, cx, cy):
    slice = im.copy()
    x, y = slice.shape
    y1 = (cy - (ny//2))
    y2 = (cy + (ny//2))
    x1 = (cx - (nx//2))
    x2 = (cx + (nx//2))

    if y1 < 0:
        slice = np.append(np.zeros((x,abs(y1))),slice, axis=1)
        x, y = slice.shape
        y1=0
    if x1 < 0:
        slice = np.append(np.zeros((abs(x1),y)),slice, axis=0)
        x, y = slice.shape
        x1=0
    if y2 > 525:
        slice = np.append(slice, np.zeros((x,y2-512)), axis=1)
        x, y = slice.shape
    if x2 > 525:
        slice = np.append(slice, np.zeros((x2-512,y)), axis=0)
        
    slice_cropped = slice[x1:x1+256, y1:y1+256]
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
    
    
#click event function
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        centrX.append(y)
        centrY.append(x)
        font = cv2.FONT_HERSHEY_SIMPLEX
        strXY = str(x)+", "+str(y)
        cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
        cv2.imshow("image", img)
        cv2.destroyAllWindows()
    
    
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
