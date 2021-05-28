"""
Created on Fri Nov 20 10:30:19 2020

@author: Marco Penso
"""

import os
import logging

'''
conto numero img png8
'''

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def count_data(input_folder):
    A_path = os.path.join(input_folder, 'trainA', 'png_8')
    B_path = os.path.join(input_folder, 'trainB', 'png_8')

    countA = 0
    countB = 0
    logging.info('ARTEFATTI')
    for paz in sorted(os.listdir(A_path)):
        paz_path = os.path.join(A_path, paz)
        num_file = len([file for file in os.listdir(paz_path)])
        logging.info('Paz: %s, num_img %d' % (paz, num_file))
        countA = countA + num_file

    logging.info('NORMALI')
    for paz in sorted(os.listdir(B_path)):
        paz_path = os.path.join(B_path, paz)
        num_file = len([file for file in os.listdir(paz_path)])
        logging.info('Paz: %s, num_img %d' % (paz, num_file))
        countB = countB + num_file

    logging.info('Tot img A: %d, tot img B %d' % (countA, countB))
    logging.info('Num paz A: %d, Num paz B: %d' % (len(os.listdir(A_path)), len(os.listdir(B_path))))
    logging.info('diff imgs: %d' % abs(countA-countB))

if __name__ == '__main__':
    # Paths settings
    input_folder = 'F:/data3'

    d = count_data(input_folder)
