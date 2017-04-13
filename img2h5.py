#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:38:23 2017

@author: yidarvin
"""

import h5py
import numpy as np
import os
from os import listdir,mkdir
from os.path import isdir,join
import scipy.misc

# Variables
size = 224

path_home = os.getcwd()
path_data = join(path_home, 'original_images')
path_imgs_tr0 = join(path_data, 'non-glasses-training')
path_imgs_tr1 = join(path_data, 'glasses-training')
path_imgs_te0 = join(path_data, 'non-glasses-testing')
path_imgs_te1 = join(path_data, 'glasses-testing')

path_save = join(path_home, 'h5_data')
path_save_tr = join(path_save, 'training')
path_save_te = join(path_save, 'testing')

def img2h5(path_imgs, path_save, label):
    """
    Reads in all images in path.  Converts them
    to .h5 files in the correct format, and
    saves them.
    INPUTS:
    - path_imgs: (string) path of images
    - path_save: (string) path to save
    - label: (int) label of image
    OUTPUT:
    - n/a
    """
    list_imgs = listdir(path_imgs)
    for name_img in list_imgs:
        # Checking filepaths
        if name_img[0] == '.':
            continue
        if name_img[-4:] != '.jpg':
            continue
        path_img = join(path_imgs, name_img)
        name_pat = name_img.split('.')[0]
        # Reading in the image and normalizing.
        img = scipy.misc.imread(path_img)
        img = scipy.misc.imresize(img, [size,size])
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        img = img.reshape([size,size,1])
        img = img.astype(np.float32)
        img -= np.min(img)
        img /= np.max(img)
        # Saving the image as formatted h5.
        path_save_img = join(path_save, name_pat)
        if not isdir(path_save_img):
            mkdir(path_save_img)
        path_h5 = join(path_save_img, name_pat+'.h5')
        h5f = h5py.File(path_h5, 'w')
        h5f.create_dataset('data', data=img)
        h5f.create_dataset('label', data=label)
    return 0

img2h5(path_imgs_tr0, path_save_tr, 0)
img2h5(path_imgs_tr1, path_save_tr, 1)
img2h5(path_imgs_te0, path_save_te, 0)
img2h5(path_imgs_te1, path_save_te, 1)