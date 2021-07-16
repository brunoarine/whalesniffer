# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 10:49:40 2015

@author: soldeace
"""
import numpy as np

def normalize_RGBratio(image, method='sqrt'):
    if method == 'sum':    
        res = image.copy()
        mean = image.mean()
        res[:,:,1] = image[:,:,1]/image[:,:,0]*mean
        res[:,:,2] = image[:,:,2]/image[:,:,0]*mean
        res[:,:,0] = mean
        return res
    if method == 'sqrt':
        res = image.copy().astype('float')
        R = image[:,:,0].astype('float')
        G = image[:,:,1].astype('float')
        B = image[:,:,2].astype('float')
        factor = np.sqrt(R**2 + G**2 + B**2)
        res[:,:,0] = R/factor
        res[:,:,1] = G/factor
        res[:,:,2] = B/factor
        return res
    
def scaler(image):
    image = np.array(image)
    return (image.astype('float') - image.min())/(image.max() - image.min())