# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 21:22:43 2015

@author: Usuario
"""

from matplotlib.image import imread
import numpy as np

PATH = 'E:\\Temp\\baleias\\'

classes = [0,1]

def get_features(img):
    used_img = img[:,:,0]
    (m,n) = used_img.shape    
    glcm = greycomatrix(used_img,
                        [m*n/30000],
                        [0,pi/4,pi/2,3*pi/4],
                        256,
                        symmetric=True,
                        normed=True)
    features = ['ASM','correlation', 'homogeneity', 'contrast']
    props = []
    for feature in features:    
        props.append(greycoprops(glcm, feature).mean())
    return props


table = []
for category in classes:
    folder = PATH+'{}\\'.format(category)
    for filename in os.listdir(folder):
     if os.path.isfile(folder + filename):
         img =  imread(folder + filename)        
         new_line = np.concatenate(([category],get_features(img)))
         table.append(new_line)

np.savetxt('glcmtable.txt',table)