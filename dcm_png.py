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
import tqdm 
import imgaug
import pypng
import pillow

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


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
    
    
def deletefolder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
        return True
    return False


def prepare_data(input_folder):

    trainA_path = os.path.join(input_folder, 'trainA')
    trainB_path = os.path.join(input_folder, 'trainB')
    
    dcmA_path = os.path.join(trainA_path, 'dicom')
    dcmB_path = os.path.join(trainB_path, 'dicom')
    
    pngA_path = os.path.join(trainA_path, 'png')
    pngB_path = os.path.join(trainB_path, 'png')
    
    deletefolder(pngA_path)
    deletefolder(pngB_path)
    
    makefolder(pngA_path)
    makefolder(pngB_path)
    
    for pazA, pazB in zip(sorted(os.listdir(dcmA_path)), sorted(os.listdir(dcmB_path))):
    
        pazA_path = os.path.join(dcmA_path, pazA)
        pazB_path = os.path.join(dcmB_path, pazB)

        for seriesA, seriesB in zip(os.listdir(pazA_path), os.listdir(pazB_path)):

            download_locationA = os.path.join(pngA_path, pazA)
            makefolder(download_locationA)
            download_locationB = os.path.join(pngB_path, pazB)
            makefolder(download_locationB)
            
            foldA = os.path.join(pazA_path, seriesA)
            foldB = os.path.join(pazB_path, seriesB)

            logging.info('FoldA Paz: %s' % pazA)
            logging.info('Number of file: %d' % len(os.listdir(foldA)))
            logging.info('FoldB Paz: %s' % pazB)
            logging.info('Number of file: %d' % len(os.listdir(foldB)))
                    
            for file in sorted(os.listdir(foldA)):
                
                fn = file.split('.dcm')
                dcmPath = os.path.join(foldA, file)
                data_row_img = pydicom.dcmread(dcmPath)
                image = np.uint8(data_row_img.pixel_array)
                Image.fromarray(image).save(os.path.join(download_locationA, fn[0] + '.png'))

            for file in sorted(os.listdir(foldB)):
                
                fn = file.split('.dcm')
                dcmPath = os.path.join(foldB, file)
                data_row_img = pydicom.dcmread(dcmPath)
                image = np.uint8(data_row_img.pixel_array)
                Image.fromarray(image).save(os.path.join(download_locationB, fn[0] + '.png'))
    

def load_data (input_folder,
               force_overwrite=True):
    
    logging.info('input folder:')
    logging.info(input_folder)
    logging.info('................................................')
    logging.info('Converting Dicom to PNG...')
    logging.info('................................................')
    
    foldA = os.path.join(input_folder, 'trainA', 'png')
    foldB = os.path.join(input_folder, 'trainB', 'png')
    
    if not os.path.exists(foldA) or not os.path.exists(foldB) or force_overwrite:
        logging.info('Dicom files have not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder)
    
    else:
        logging.info('Already preprocessed Dicom files')


if __name__ == '__main__':
    
    # Paths settings
    input_folder = '/content/drive/My Drive/PACEmaker/data'
        
    d=load_data(input_folder)
