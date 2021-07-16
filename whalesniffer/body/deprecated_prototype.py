# -*- coding: utf-8 -*-

from matplotlib.image import imread
from matplotlib.pyplot import imshow, scatter, plot, colorbar
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import rank
from skimage.filters import sobel
from skimage.feature import canny
from skimage.filters.thresholding import threshold_otsu, threshold_local
from scipy.signal import convolve2d
import numpy as np
import operator
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.morphology import skeletonize, binary_erosion, dilation, reconstruction
from scipy import ndimage as ndi
from scipy import fftpack
from skimage.feature import local_binary_pattern
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import rotate
from skimage import exposure
from itertools import permutations
from skimage.measure import moments_hu
from skimage.feature import match_template
from pylab import figure, title
import cv2
from skimage import transform as tf
import cv2
from pylab import *
import colorsys
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from skimage import morphology
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage as ndi
from ..utils import blob
from ..utils import color
import mahotas

#################################################################
def estimate_noise(I, mode='sobel'):  
    (h,w) = I.shape
    if mode == 'sobel':                
        sigma = np.sum(sobel(I))/float(h*w)        
    elif mode == 'fft':
        F1 = fftpack.fft2(I)
        F2 = fftpack.fftshift(F1)
        psd2D = np.abs( F2 )**2
        logimg = np.log(psd2D)
        sigma = np.std(logimg[h/2-20:h/2+20, w/2-20:w/2+20])
    return sigma

def denoise(img, mode=rank.median):    
    (h,w) = img.shape
    blur_shape = (int(h/20.), int((h/20.)))
    output = mode(img, np.ones(blur_shape))
    return output
    


class Prototype:
    def predict(self, filenames_list):
        if not isinstance(filenames_list, list):
            raise Exception('Input list of files is not a list actually')
        
        rectangles = []
        for filename in filenames_list:       
            
            img_rgb = imread(filename)
          
            img_hsv = rgb2hsv(img_rgb)
            img_r = img_rgb[:,:,0]
            img_g = img_rgb[:,:,1]
            img_b = img_rgb[:,:,2]
            img_v = img_hsv[:,:,2]
            img_s = img_hsv[:,:,1]
            img_h = img_hsv[:,:,0]
            #img_h[img_h < 0.4] = 1
            img_rgbn = color.normalize_RGBratio(img_rgb, method='sqrt')
            img_rn = img_rgbn[:,:,0]
            img_lab = rgb2lab(img_rgb)
            img_a = img_lab[:,:,1]
            img_c = 1-np.arctan(img_rgb[:,:,2]/np.maximum(img_rgb[:,:,1],img_rgb[:,:,0]))
            
                 
            
            h,w = img_r.shape
            #############################################################
            # Which channel is the smoothest?
            #############################################################  
            
            candidates = [img_h, img_a, img_rn] 
            noises = []
#            for idx, channel in enumerate(candidates):    
#                noise_value = estimate_noise(channel, mode='sobel')    
#                noises.append((idx, noise_value))
#            img_ref = candidates[min(noises, key=operator.itemgetter(1))[0]]
            img_ref = color.scaler((color.scaler(img_a) + 0.1) * (color.scaler(img_rn) + 0.1) + (color.scaler(img_c) + 0.2))
            #(color.scaler(img_c) + 0.1))
            #(color.scaler(img_rn) + 0.1)))
                      
            #############################################################
            # Remove noise    
            #############################################################
            img_smoothed = denoise(img_ref, mode=rank.median)
            segmented = color.scaler(rank.gradient(img_smoothed, disk(h*w/55756)))
            
            
            #############################################################
            # THRESHOLDING
            #############################################################
#            threshold = threshold_otsu(segmented)
#            binary = segmented > threshold
            binary = segmented > 0.2
           
        
            #############################################################
            #Closing
            ###################
            #closed = binary_closing(binary, disk(30))
            img_closed = mahotas.close(binary, disk(3))
            img_fill = ndi.binary_fill_holes(img_closed)
          
            #############################################################
            # SMALLER BLOBS REMOVAL
            #############################################################
            label_objects, nb_labels = ndi.label(img_fill)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes == np.sort(sizes)[-2]
            img_cleaned = mask_sizes[label_objects]
           
            
            
    
            
            
        
            print(filename)
            rectangles.append(blob.bound_rect(img_cleaned))
        return rectangles
