# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:48:27 2015

@author: Usuario
"""
from skimage.filters import rank, sobel, gaussian_filter
from skimage.feature import canny
import numpy as np
import operator
from scipy import fftpack
from pylab import figure, imshow

def denoise(img, mode=rank.median):    
    (h,w) = img.shape
    blur_shape = (int(h/20.), int((h/20.)))
    output = mode(img, np.ones(blur_shape))
    return output
    
def least_noise(candidates):
    noises = []
    for idx, channel in enumerate(candidates):
        channel_clean = gaussian_filter(channel, 16)
        noise_value = estimate_noise(channel_clean, mode='sobel')    
        noises.append((idx, noise_value))    
    img_ref = candidates[min(noises, key=operator.itemgetter(1))[0]]
    return img_ref

def estimate_noise(I, mode='sobel'):  
    (h,w) = I.shape
    if mode == 'sobel':        
        sigma = np.sum(sobel(I))/float(h*w) 
    if mode == 'canny':
        sigma = np.sum(canny(I))/float(h*w)        
    elif mode == 'fft':
        F1 = fftpack.fft2(I)
        F2 = fftpack.fftshift(F1)
        psd2D = np.abs( F2 )**2
        logimg = np.log(psd2D)
        sigma = np.std(logimg[h/2-20:h/2+20, w/2-20:w/2+20])
    return sigma
