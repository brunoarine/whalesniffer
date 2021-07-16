# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 22:21:15 2015

@author: Usuario
"""

import numpy as np
import cv2
from scipy import ndimage as ndi

def getHist(im):
    ### Get the histogram. All histograms must be the same kind
    hist = cv2.calcHist([im],[0,1,2],None,[16,16,16],[0,255,0,255,0,255])
    return(hist)


def divImage(im):
    ### Divide the image in four subimages
    height,width,channels = im.shape
    im1 = im[0:height//2, 0:width//2]
    im2 = im[0:height//2, width//2:width]
    im3 = im[height//2:height, 0:width//2]
    im4 = im[height//2:height, width//2:width]
    return([im1,im2,im3,im4])


def setImage(im,val=0):
    ### Sets a whole image to a value. By default, black
    for i in range(0,3):
        im[:,:,i]=val

def simHist(im, baseHist, stop):
    '''
    Create a mask image to select regions with a distinct histogram. 
      white zones are different enough from the base image
      im: image
      baseHist: Histogram of the base image (currently, the original image)
      label: A label for the current step, just for debug 
    '''

    height,width,channels = im.shape
    if width < stop or height < stop:
        setImage(im,0)
        return
    images = divImage(im)
    histg = baseHist
    
    ### Test
    # This compares each subimage with the current image's histogram
    # instead of the base (whole image)
    # histg = getHist(im)
    # CORREL: Other similarity measures may be tested
    sim = [cv2.compareHist(histg.flatten(), getHist(imx).flatten(),
                           cv2.HISTCMP_CORREL) for imx in images]
    for i in range(0,len(sim)):
        #This is the threshold to consider too different 
        if sim[i] < 0.25:
            setImage( images[i], 255 )
        else:
            simHist(images[i], histg, stop)
    return