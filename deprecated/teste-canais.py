# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 19:32:28 2015

@author: Usuario
"""

from matplotlib.image import imread
from matplotlib.pyplot import imshow, scatter, plot, colorbar, figure
import numpy as np
from skimage.color import rgb2hsv
from skimage.segmentation import felzenszwalb, slic, quickshift
from skimage.transform import resize

def show(img, cmap='viridis'):
    figure(figsize=(7.5,5))    
    imshow(img, cmap=cmap)
    colorbar();

filename = 'C:\\Users\\Usuario\\Downloads\\imgs_subset\\imgs_subset\\w_148.jpg'
img_rgb = imread(filename)

img_hsv = rgb2hsv(img_rgb)
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]
img_v = img_hsv[:,:,2]
img_s = img_hsv[:,:,1]
img_h = img_hsv[:,:,0]
O1 = (img_r- img_g)/np.sqrt(2)
O2 = (img_r+ img_g - 2*img_b)/np.sqrt(6)
O3 = (img_r+ img_g+ img_b)/np.sqrt(3)

img_o = img_rgb.copy()
img_o[:,:,0] = O1
img_o[:,:,1] = O2
img_o[:,:,2] = O3

norm = np.sqrt(img_r.astype('float')**2 + img_g.astype('float')**2 + img_b.stype('float')**2)
N1 = img_r.astype('float')/norm
N2 = img_g.astype('float')/norm
N3 = img_b.astype('float')/norm

img_n = img_rgb.copy()
img_n[:,:,0] = (N1*255).astype('uint8')
img_n[:,:,1] = (N2*255).astype('uint8')
img_n[:,:,2] = (N3*255).astype('uint8')


show(img_o)

def normalize_RGBratio(image):
    res = image.copy()
    mean = image.mean()
    print mean
    res[:,:,1] = image[:,:,1]/image[:,:,0]*mean
    res[:,:,2] = image[:,:,2]/image[:,:,0]*mean
    res[:,:,0] = mean
    return res