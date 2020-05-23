import os
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
import matplotlib.pyplot as plt
import sys
import shutil
import png
import itertools
import pydicom # for reading dicom files
import pandas as pd # for some simple data analysis (right now, just to load in the labels data and quickly reference it)
import imgaug

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
cv2.destroyAllWindows()
centrX = []
centrY = []

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
            

def prepare_data(input_folder):

    cv2.destroyAllWindows()
    centrX = []
    centrY = []
    
    #click event function
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            #print(x,",",y)
            centrX.append(y)
            centrY.append(x)
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x)+", "+str(y)
            cv2.putText(img, strXY, (x,y), font, 0.5, (255,255,0), 2)
            cv2.imshow("image", img)
            cv2.destroyAllWindows()
    
    
    trainA_path = os.path.join(input_folder, 'trainA')
    trainB_path = os.path.join(input_folder, 'trainB')
    
    pngA_path = os.path.join(trainA_path, 'png')
    pngB_path = os.path.join(trainB_path, 'png')
    
    cropA_path = os.path.join(trainA_path, 'crop_png')
    cropB_path = os.path.join(trainB_path, 'crop_png')
    
    deletefolder(cropA_path)
    deletefolder(cropB_path)
    
    makefolder(cropA_path)
    makefolder(cropB_path)
    
    for pazA, pazB in zip(sorted(os.listdir(pngA_path)), sorted(os.listdir(pngB_path))):
    
        pazA_path = os.path.join(pngA_path, pazA)
        pazB_path = os.path.join(pngB_path, pazB)

        download_locationA = os.path.join(cropA_path, pazA)
        makefolder(download_locationA)
        download_locationB = os.path.join(cropB_path, pazB)
        makefolder(download_locationB)
        
        
        logging.info('Processing Paz: %s' % pazA)
        addr_img = []
        name_img = []
        
        for file in sorted(os.listdir(pazA_path)):
            addr = os.path.join(pazA_path, file)
            addr_img.append(addr)
            name_img.append(file)
            
        num_file = len(addr_img)
        median_centrX = []
        median_centrY = []
        
        for phase in range(30):
            n_frame = len(range(phase, num_file, 30))
            centrX = []
            centrY = []
            if n_frame % 2 == 0:
                var = int(n_frame/2)
                frames = [phase+(30*(var-1)), phase+(30*var)]
            elif n_frame % 2 != 0:
                var = int(n_frame/2)
                frames = [phase+(30*(var-1)), phase+(30*var), phase+(30*(var+1))]

            for frame in frames:
                img = np.array(Image.open(addr_img[frame])).astype("uint16")
                img = cv2.normalize(img, dst=None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
                img = img.astype("uint8")
                cv2.imshow("image", img)
                cv2.namedWindow('image')
                cv2.setMouseCallback("image", click_event)
                cv2.waitKey(0)  
            median_centrX.append(int(np.median(centrX)))
            median_centrY.append(int(np.median(centrY)))
        
        logging.info('Saving Data...')
        for phase in range(30):
            for frame in range(phase, num_file, 30):
                im = np.array(Image.open(addr_img[frame])).astype("uint16")
                im2 = im.copy()
                im2 = crop_or_pad_slice_to_size(im, 256, 256, median_centrX[phase], median_centrY[phase])
                array_buffer = im2.tobytes()
                img = Image.new("I", im2.shape)
                img.frombytes(array_buffer, 'raw', "I;16")
                img.save(os.path.join(download_locationA, name_img[frame]))

        
        logging.info('Processing Paz: %s' % pazB)
        addr_img = []
        name_img = []
        
        for file in sorted(os.listdir(pazB_path)):
            addr = os.path.join(pazB_path, file)
            addr_img.append(addr)
            name_img.append(file)
        
        num_file = len(addr_img)
        median_centrX = []
        median_centrY = []
        
        for phase in range(30):
            n_frame = len(range(phase, num_file, 30))
            centrX = []
            centrY = []
            if n_frame % 2 == 0:
                var = int(n_frame/2)
                frames = [phase+(30*(var-1)), phase+(30*var)]
            elif n_frame % 2 != 0:
                var = int(n_frame/2)
                frames = [phase+(30*(var-1)), phase+(30*var), phase+(30*(var+1))]

            for frame in frames:
                img = np.array(Image.open(addr_img[frame])).astype("uint16")
                img = cv2.normalize(img, dst=None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
                img = img.astype("uint8")
                cv2.imshow("image", img)
                cv2.namedWindow('image')
                cv2.setMouseCallback("image", click_event)
                cv2.waitKey(0)  
            median_centrX.append(int(np.median(centrX)))
            median_centrY.append(int(np.median(centrY)))
        
        logging.info('Saving Data...')
        for phase in range(30):
            for frame in range(phase, num_file, 30):
                im = np.array(Image.open(addr_img[frame])).astype("uint16")
                im2 = im.copy()
                im2 = crop_or_pad_slice_to_size(im, 256, 256, median_centrX[phase], median_centrY[phase])
                array_buffer = im2.tobytes()
                img = Image.new("I", im2.shape)
                img.frombytes(array_buffer, 'raw', "I;16")
                img.save(os.path.join(download_locationB, name_img[frame]))



def load_data (input_folder,
               force_overwrite=True):
    
    logging.info('input folder:')
    logging.info(input_folder)
    logging.info('................................................')
    
    foldA = os.path.join(input_folder, 'trainA', 'crop_png')
    foldB = os.path.join(input_folder, 'trainB', 'crop_png')
    
    if not os.path.exists(foldA) or not os.path.exists(foldB) or force_overwrite:
        logging.info('PNG files have not yet been cropped')
        logging.info('Preprocessing now!')
        prepare_data(input_folder)
    
    else:
        logging.info('Already cropped PNG files')


if __name__ == '__main__':
    
    # Paths settings
    input_folder = 'F:/prova/data'
        
    d=load_data(input_folder)
