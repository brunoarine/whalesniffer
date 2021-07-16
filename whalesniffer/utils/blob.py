# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:42:07 2015

@author: Usuario
"""

from scipy import ndimage as ndi
import numpy as np

def largest_blob(img):
    label_objects, nb_labels = ndi.label(img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes == np.sort(sizes)[-2]
    clean_img = mask_sizes[label_objects]
    return clean_img
    
def bound_rect(img):    
    crop = np.nonzero(img)
    return np.array([[crop[0].min(), crop[1].min()],
                      [crop[0].max(), crop[1].max()]])