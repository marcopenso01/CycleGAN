"""
Created on Fri Feb 19 11:39:50 2021

@author: Marco Penso
"""

import os
import numpy as np
import logging
import cv2
import shutil
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance

drawing=False # true if mouse is pressed
mode=True

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

# mouse callback function
def paint_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y
    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),5)
                cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),5)
                current_former_x = former_x
                current_former_y = former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),5)
            cv2.line(image_binary,(current_former_x,current_former_y),(former_x,former_y),(255,255,255),5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y
  
def imfill(img):
    img = img[:,:,0]
    img = cv2.resize(img, (200, 200))
    img[img>0]=255
    im_floodfill = img.copy()
    h, w = im_floodfill.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    return img | cv2.bitwise_not(im_floodfill)

if __name__ == '__main__':
    # Paths settings
    input_folder = r'F:\prova\trainA'
    paz = 'pazpaz'
    force_overwrite=True
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('Genero maschere:')
    logging.info('................................................')
    mask_path = os.path.join(input_folder, 'maschere')
    paz_out = os.path.join(mask_path, paz)
    input_path = os.path.join(input_folder, 'resize8bit')
    
    if not os.path.exists(mask_path):
        makefolder(mask_path)
        
    if force_overwrite:
        if os.path.exists(paz_out):
            deletefolder(paz_out)
        makefolder(paz_out)
        
    paz_path = os.path.join(input_path, paz)
    num_file = len([file for file in os.listdir(paz_path)])
    logging.info('Paz: %s, num_img %d' % (paz, num_file))

    #makefolder(os.path.join(paz_out,'artefact'))
    makefolder(os.path.join(paz_out,'ventricle DX'))
    #makefolder(os.path.join(paz_out,'ventricle SX'))
    #makefolder(os.path.join(paz_out,'final_seg'))
    tit=['ventricle DX contour']
    for file in sorted(os.listdir(paz_path)):
        addr = os.path.join(paz_path, file)       
        for ii in range(1):
            img = cv2.imread(addr)
            #img = Image.open(addr)
            clahe = cv2.createCLAHE(clipLimit = 2)
            img = clahe.apply(img[:,:,0])
            #enhancer = ImageEnhance.Brightness(img)
            #factor = 2 #increase contrast
            #img = enhancer.enhance(factor)
            #enhancer2 = ImageEnhance.Contrast(img)
            #img = enhancer2.enhance(factor)
            
            img = np.array(img)
            img = cv2.resize(img, (400, 400))  
            image_binary = np.zeros((img.shape[1], img.shape[0], 1), np.uint8)
            cv2.namedWindow(tit[ii])
            cv2.setMouseCallback(tit[ii],paint_draw)
            while(1):
                cv2.imshow(tit[ii],img)
                k=cv2.waitKey(1)& 0xFF
                if k==27: #Escape KEY
                    im_out1 = imfill(image_binary)
                    im_out1[im_out1>0]=255
                    cv2.imwrite(os.path.join(paz_out, 'ventricle DX', file), im_out1)                    
                    break              
            cv2.destroyAllWindows()
