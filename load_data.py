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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def standardize_image(image):
    '''
    make image zero mean and unit standard deviation
    '''

    img_o = np.float32(image.copy())
    m = np.mean(img_o)
    s = np.std(img_o)
    return np.divide((img_o - m), s)
    
    
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
    
    for foldersA, foldersB in zip(sorted(os.listdir(trainA_path)), sorted(os.listdir(trainB_path))):
        
        foldA_path = os.path.join(trainA_path, foldersA)
        foldB_path = os.path.join(trainB_path, foldersB)
        
        pathA = os.path.join(foldA_path, '*.png')
        pathB = os.path.join(foldB_path, '*.png')
        
        for fileA in sorted(glob.glob(pathA)):
            trainA_addrs.append(fileA)
        for fileB in sorted(glob.glob(pathB)):
            trainB_addrs.append(fileB)
    
    trainA_shape = (len(trainA_addrs), nx, ny)
    trainB_shape = (len(trainB_addrs), nx, ny)
    
    hdf5_file.create_dataset("trainA", trainA_shape, dtype=np.float32)
    hdf5_file.create_dataset("trainB", trainB_shape, dtype=np.float32)
    
    for i in range(len(trainA_addrs)):
        addr_img = trainA_addrs[i]
        img = cv2.imread(addrs_img, 0)
        img = standardize_image(img)
        img = cv2.resize(img, (nx,ny), interpolation=cv2.INTER_AREA)
        hdf5_file["trainA"][i, ...] = img[None]
    
    for i in range(len(trainB_addrs)):
        addr_img = trainB_addrs[i]
        img = cv2.imread(addrs_img, 0)
        img = standardize_image(img)
        img = cv2.resize(img, (nx,ny), interpolation=cv2.INTER_AREA)
        hdf5_file["trainB"][i, ...] = img[None]    
    
    hdf5_file.close()
    
    
def load_data (input_folder,
               preprocessing_folder,
               size,
               force_overwrite=True):
    
    size_str = '_'.join([str(i) for i in size])
    file_name = 'data_2D_size_%s.hdf5' % (size_str)
    file_path = os.path.join(preprocessing_folder, file_name)
    makefolder(preprocessing_folder)
    
    if not os.path.exists(file_path) or force_overwrite:
        logging.info('This configuration has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, output_file, size)
    
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
