# -*- coding: utf-8 -*-

import cv2
import mahotas
import numpy as np
from ..utils import blob, hist, color


class Similarity():
    '''
    Detects whales through histogram similarity. The function stores the image
    histogram as reference and splits the image in four parts. It iterates
    through each part to evaluate histogram similarity with the reference
    histogram. If no correlation is found, the region is probably a distinct
    object from the scene and is therefore marked. Otherwise, it recursively
    divides this region in four sub-regions for a new iteration and so on.
    
    Parameters
    ----------
    
    metric : string, optional
        Used metric to evaluate how different a sub-region is from the rest
        of the image. Available option is only 'corr' for the moment.
    stop : integer, optional
        Minimum region size for iteration breakpoint (in pixels).
        Default value is 2.
    '''
    
    def __init__(self, metric='corr',colorspace='hsv',stop=2):        
        self.metric = metric
        self.colorspace = colorspace
        self.stop = stop
    
    def predict(self, filenames_list):
        if not isinstance(filenames_list, list):
            raise Exception('Input list of files is not a list actually')
        
        rectangles = []
        for filename in filenames_list:    
            im = cv2.imread(filename)
            (h,w) = im[:,:,0].shape
            ### PREPROCESS
            # Convert to HSV, yields better results than BGR
            if self.colorspace == 'hsv':
                im = cv2.cvtColor(im,cv2.COLOR_BGR2HSV)
            elif self.colorspace == 'lab':
                im = cv2.cvtColor(im,cv2.COLOR_BGR2LAB)
            elif self.colorspace == 'rgbn':
                im = (color.normalize_RGBratio(im)*255).astype('uint8')
            # gets original image hist
            baseHist = hist.getHist(im)
            
            ### PROCESS
            # Perform region selection by histogram similarity.
            # Returns a mask on whole image
            hist.simHist(im,baseHist, stop=self.stop)           
            
            ### POST
            # Erase smaller detected blobs
            kernel = np.ones((h//51,h//51),np.uint8)
            im = mahotas.erode(im[:,:,0], kernel)
            im = mahotas.dilate(im, kernel)            
            im = blob.largest_blob(im)
            
            ### Draw boundary rectangle
            rectangles.append(blob.bound_rect(im))
                                            
        return rectangles
    
