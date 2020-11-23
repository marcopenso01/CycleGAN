"""
Created on Tue Jul 14 09:13:34 2020

@author: Marco Penso
"""

import os
import numpy as np
import logging
import h5py
from skimage import transform
import cv2
from PIL import Image
import shutil
import pydicom  # for reading dicom files
import math as mt

import Read_dicom

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
cv2.destroyAllWindows()
X = []
Y = []


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


def rotate_image(img, angle, interp=cv2.INTER_LINEAR):
    rows, cols = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, rotation_matrix, (cols, rows), flags=interp)


def translate_image(img, px, py):
    M = np.float32([[1, 0, px], [0, 1, py]])
    return cv2.warpAffine(img, M, (img.shape[0], img.shape[1]))


def normalize_image(image):
    '''
    make image normalize between 0 and 1  (OR -1,1)
    '''
    img_o = np.float32(image.copy())
    # img_o = 2*((img_o-img_o.min())/(img_o.max()-img_o.min()))-1
    img_o = (img_o - img_o.min()) / (img_o.max() - img_o.min())
    return img_o

def normalize_image2(image):
    # If using 16 bit depth images, use the formula 'array = array / 32767.5 [0-1]
    img_o = np.float32(image.copy())
    img_o = img_o / 32767.5
    return img_o


def crop_or_pad_slice_to_size_specific_point(im, nx, ny, cx, cy):
    slice = im.copy()
    x, y = slice.shape
    y1 = (cy - (ny // 2))
    y2 = (cy + (ny // 2))
    x1 = (cx - (nx // 2))
    x2 = (cx + (nx // 2))

    if y1 < 0:
        slice = np.append(np.zeros((x, abs(y1))), slice, axis=1)
        x, y = slice.shape
        y1 = 0
    if x1 < 0:
        slice = np.append(np.zeros((abs(x1), y)), slice, axis=0)
        x, y = slice.shape
        x1 = 0
    if y2 > 525:
        slice = np.append(slice, np.zeros((x, y2 - 512)), axis=1)
        x, y = slice.shape
    if x2 > 525:
        slice = np.append(slice, np.zeros((x2 - 512, y)), axis=0)

    slice_cropped = slice[x1:x1 + 256, y1:y1 + 256]
    return slice_cropped


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


def prepare_data(input_folder, train_fold, nx, ny):
    cv2.destroyAllWindows()
    X = []
    Y = []

    # click event function
    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # print(x,",",y)
            X.append(y)
            Y.append(x)
            font = cv2.FONT_HERSHEY_SIMPLEX
            strXY = str(x) + ", " + str(y)
            cv2.putText(img, strXY, (x, y), font, 0.5, (255, 255, 0), 2)
            cv2.imshow("image", img)
            cv2.destroyAllWindows()

    train_path = os.path.join(input_folder, train_fold)

    dcmA_path = os.path.join(input_folder, 'trainA', 'dicom')
    dcmB_path = os.path.join(input_folder, 'trainB', 'dicom')

    png_path = os.path.join(train_path, 'png')

    png8_path = os.path.join(train_path, 'png_8')

    proc_path = os.path.join(train_path, 'preprocessing')

    if os.path.exists(proc_path):
        deletefolder(proc_path)

    makefolder(proc_path)

    data_file_path = os.path.join(proc_path, 'train.hdf5')
    hdf5_file = h5py.File(data_file_path, "w")

    Pixel_sizeXa = []
    Pixel_sizeYa = []
    Pixel_sizeXb = []
    Pixel_sizeYb = []

    # salvo pixel size di ogni paziente
    for pazA, pazB in zip(sorted(os.listdir(dcmA_path)), sorted(os.listdir(dcmB_path))):
        pazA_path = os.path.join(dcmA_path, pazA)
        pazB_path = os.path.join(dcmB_path, pazB)

        for seriesA, seriesB in zip(os.listdir(pazA_path), os.listdir(pazB_path)):
            foldA = os.path.join(pazA_path, seriesA)
            foldB = os.path.join(pazB_path, seriesB)

            dcmPath = os.path.join(foldA, os.listdir(foldA)[0])
            data_row_img = pydicom.dcmread(dcmPath)
            Pixel_sizeXa.append(data_row_img.PixelSpacing[0])
            Pixel_sizeYa.append(data_row_img.PixelSpacing[1])

            dcmPath = os.path.join(foldB, os.listdir(foldB)[0])
            data_row_img = pydicom.dcmread(dcmPath)
            Pixel_sizeXb.append(data_row_img.PixelSpacing[0])
            Pixel_sizeYb.append(data_row_img.PixelSpacing[1])

    target_resolutionX = min(min(Pixel_sizeXa), min(Pixel_sizeXb))
    target_resolutionY = min(min(Pixel_sizeYa), min(Pixel_sizeYb))

    #imposto dimensione dataset hdf5
    n_file = 0
    for paz in sorted(os.listdir(png8_path)):
        paz_path = os.path.join(png8_path, paz)

        n_file = n_file + len(os.listdir(paz_path))

    logging.info('train images: %d' % n_file)

    train_shape = (n_file, nx, ny)
    

    hdf5_file.create_dataset("images_train", train_shape, np.float32)
    hdf5_file.create_dataset("path", (n_file,), dtype=h5py.special_dtype(vlen=str))
    hdf5_file.create_dataset("angle", (n_file,), np.float32)
    hdf5_file.create_dataset("transY", (n_file,), np.float32)
    hdf5_file.create_dataset("transX", (n_file,), np.float32)
    hdf5_file.create_dataset("scale", (n_file,2), np.float32)

    #conto tot file png
    tot_file = []
    nn = 0
    for paz in sorted(os.listdir(png_path)):
        paz_path = os.path.join(png_path, paz)
        tot_file.append(len(os.listdir(paz_path)))

    #pre-process file
    for paz, i in zip(sorted(os.listdir(png8_path)), range(len(os.listdir(png8_path)))):

        logging.info('--------------------------------------------------------------')
        logging.info(i)
        logging.info(train_fold)
        logging.info('Processing Paz: %s' % paz)
        train_addrs = []
        all_addrs = []
        if train_fold == 'trainA':
            scale_vector = [Pixel_sizeXa[i] / target_resolutionX, Pixel_sizeYa[i] / target_resolutionY]
        elif train_fold == 'trainB':
            scale_vector = [Pixel_sizeXb[i] / target_resolutionX, Pixel_sizeYb[i] / target_resolutionY]
        print('scale_vector: %s' % scale_vector)

        for file in sorted(os.listdir(os.path.join(png8_path, paz))):
            addr = os.path.join(png_path, paz, file)
            train_addrs.append(addr)
        for file in sorted(os.listdir(os.path.join(png_path, paz))):
            addr = os.path.join(png_path, paz, file)
            all_addrs.append(addr)

        angles = []
        translX = []
        translY = []

        for phase in range(30):
            n_frame = len(range(phase, tot_file[i], 30))
            var = int(n_frame / 2)
            frame = (phase+1) + (30 * (var-1))
            X = []
            Y = []
            img = np.array(Image.open(all_addrs[frame-1])).astype("uint16")
            img = cv2.normalize(img, dst=None, alpha=0, beta=256, norm_type=cv2.NORM_MINMAX)
            img = img.astype("uint8")
            rows, cols = img.shape[:2]
            for ii in range(2):
                cv2.imshow("image", img)
                cv2.namedWindow('image')
                cv2.setMouseCallback("image", click_event)
                cv2.waitKey(0)
            '''rotate image'''
            A = [int(round(X[0])), int(round(Y[0]))]
            B = [int(round(X[1])), int(round(Y[1]))]
            ''' find point2 (point with larger x) '''
            if A[0] > B[0]:
                P2 = A
                P1 = B
            else:
                P2 = B
                P1 = A
            ''' clockwise OR counterclockwise'''
            if P2[1] < P1[1]:
                w = +1
            else:
                w = -1
            angles.append(w * mt.atan(abs(P2[1] - P1[1]) / (P2[0] - P1[0] + 0.0000000000001)) * 180 / 3.141592653589793)
            '''translate image'''
            crRows = int(abs((A[0] + B[0]) / 2))
            crCols = int(abs((A[1] + B[1]) / 2))
            if crRows > int(rows / 2):
                translX.append(int(rows / 2) - crRows)
            else:
                translX.append(int(rows / 2) - crRows)
            if crCols > int(cols / 2):
                translY.append(int(cols / 2) - crCols)
            else:
                translY.append(int(cols / 2) - crCols)

        logging.info('Saving Data...')
        for n in range(len(train_addrs)):
            file = train_addrs[n]
            num_img = int(file.split('img')[1].split('-')[0])
            phase = int(num_img % 30)
            if phase == 0:
                phase = 30
            
            #logging.info('file %s' % file)
            #logging.info('num_img %d' % num_img)
            #logging.info('phase %d' % phase)

            im = np.array(Image.open(file)).astype("float32")
            im2 = im.copy()
            #im2 = normalize_image2(im2)
            im2 = rotate_image(im2, angles[phase-1])
            
            '''y=colm, x=row'''
            im2 = translate_image(im2, translY[phase-1], translX[phase-1])
            slice_rescaled = transform.rescale(im2,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               anti_aliasing=True,
                                               mode='constant')
            slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
            hdf5_file["images_train"][nn, ...] = slice_cropped[None]
            hdf5_file["path"][nn, ...] = file
            hdf5_file["angle"][nn, ...] = angles[phase-1]
            hdf5_file["transY"][nn, ...] = translY[phase-1]
            hdf5_file["transX"][nn, ...] = translX[phase-1]
            hdf5_file["scale"][nn, ...] = np.float32(scale_vector)
            
            nn = nn + 1
            

    hdf5_file.close()



def load_data(input_folder,
              train_fold,
              force_overwrite=False,
              flag_dicom=False,
              nx = 200,
              ny = 200):
    

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('input folder:')
    logging.info(input_folder)
    logging.info('................................................')

    if flag_dicom:
        Read_dicom.load_data(input_folder)

    pngA = os.path.join(input_folder, 'trainA', 'png_8')
    pngB = os.path.join(input_folder, 'trainB', 'png_8')
    if not os.path.exists(pngA) or not os.path.exists(pngB):
        logging.warning('png_8 folder has not been found')

    n_fileA = 0
    n_fileB = 0
    for pazA, pazB in zip(sorted(os.listdir(pngA)), sorted(os.listdir(pngB))):
        pazA_path = os.path.join(pngA, pazA)
        pazB_path = os.path.join(pngB, pazB)
        n_fileA = n_fileA + len(os.listdir(pazA_path))
        n_fileB = n_fileB + len(os.listdir(pazB_path))
    if not n_fileA == n_fileB:
        logging.warning(' - Number of imgs is not equal in the png_8 folders - A: %d, - B: %d' % (n_fileA, n_fileB))

    fold = os.path.join(input_folder, train_fold, 'preprocessing')

    logging.info('................................................')
    logging.info('Preprocessing PNG files...')
    logging.info('................................................')

    if not os.path.exists(fold) or force_overwrite:
        logging.info('files have not yet been Processed')
        logging.info('Preprocessing now!')
        logging.info('path: %s' % fold)
        prepare_data(input_folder, train_fold, nx, ny)

    else:
        logging.info('Already preprocessed files')


if __name__ == '__main__':
    # Paths settings
    input_folder = 'F:/prova'
    train_fold = 'trainA'
    d = load_data(input_folder, train_fold, force_overwrite=True, flag_dicom=False, nx=200, ny=200)
    #force_overwrite = sovrascrive file
    #flag_dicom = legge file dicom se non ancora letto
    #nx, ny = dimensione img in uscita
    #folder = cartella da processare (trainA o trainB)
