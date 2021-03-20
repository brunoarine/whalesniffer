# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 18:48:04 2015

@author: Usuario
"""

import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage import morphology
from skimage.feature import greycomatrix, greycoprops
from matplotlib.image import imsave
from skimage.filters.rank import entropy
from skimage.morphology import disk
import os
from scipy import ndimage as ndi

def binarize(image, mode, disk_size):
    '''
    Applies blur through a blur_size mask, and then finds global threshold
    '''
    img_blur = cv2.blur(image, (disk_size,disk_size))
    threshold = threshold_otsu(img_blur)
    img_bw = img_blur < threshold
    if mode == 'skeletonize':
        kernel = disk(disk_size)
        img_eroded = cv2.erode(image.astype('uint8'), kernel, iterations=1)
        img_bw = cv2.dilate(img_eroded, kernel, iterations=1)        
    return img_bw
    
def fit(img_bw, mode, order=3):
    if mode == 'skeletonize':
        img_fit = morphology.skeletonize(img_bw)
    else:
        (m,n) = img_bw.shape
        (y,x) = np.nonzero(img_bw)
        table = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
        table = table[np.argsort(table[:,0])]
        x = table[:,0]
        y = table[:,1]
        coeffs = np.polyfit(x,y,3)
        yp = np.polyval(list(coeffs),x).astype('int')
        img_fit = np.zeros((m,n))
        img_fit[yp,x] = 1    
    return img_fit
    
def largest_only(img):
    label_objects, nb_labels = ndi.label(img)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes == np.sort(sizes)[-2]
    img_cleaned = mask_sizes[label_objects]
    return img_cleaned


def normalize(img):
    '''
    Tries to standardize an image.
    '''
    img = img.astype('float')
    output = 255*(img - img.min())/(img.max() - img.min())
    output = output.astype('uint8')
    return output


def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))
    
def hist(lbp):
    n_bins = lbp.max() + 1
    return plt.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')



def extract_head(filename):
    '''
    Tries to locate whale's head in aerial image. Returns rectangular array
    containing the head.
    '''    
    img_bgr = cv2.imread(filename)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_h = normalize(img_hsv[:,:,0])
    img_s = normalize(img_hsv[:,:,1])
    img_v = normalize(img_hsv[:,:,2])
    img_r = normalize(img_rgb[:,:,0])
    (h,w) = img_s.shape
    
    # First of all, a decent amount of blur must be applied before anything
    # else. However, this amount is image size dependent. I've found that
    # ImageHeight/5 works in most cases.    
    img_bw = binarize(img_s, 'blur', disk_size=h/40) # blur = h/40, open h/15
    occupied_area = sum(img_bw)/float(h*w)
    print 'Occupied area:',occupied_area
    if occupied_area > 0.4: # symptom that something went wrong
        img_entropy = entropy(img_v, disk(h/25))
        img_bw = 1-binarize(img_entropy, 'blur', disk_size=h/10)
    imshow(img_bw)
    
    # Find skeleton endpoints
    # Skeletonizing must happen with properly dilated whale through a blur
    # process. Otherwise, skeletons will spread out to irrelevant noise.
    # alternatives: skeleton = morphology.skeletonize(img_bw)
    #               skeleton = fit(img_bw, order=3)
    skeleton = fit(largest_only(img_bw), 'skeletonize', 3)
    imshow(skeleton)
        
    (i_idx,j_idx) = np.nonzero(skeleton)
    p1 = np.nonzero(skeleton[:,j_idx.min()])[0][0], j_idx.min()
    p2 = np.nonzero(skeleton[:,j_idx.max()])[0][0], j_idx.max()
    p1 = np.array(p1)
    p2 = np.array(p2)
    endpoints = np.array([p1,p2])
    
    d = int(np.linalg.norm(p2-p1)/5.5)
    
    # Now let's define an area for these found endpoints and their respective
    # texture analysis parameter to decide which endpoint is the head.
    
    data = []    
    for p in endpoints:
        (i,j) = p
        area = img_h[max(i-d,0):min(i+d,h), max(j-d,0):min(j+d,w)]     
        colored_area = img_rgb[max(i-d,0):min(i+d,h), max(j-d,0):min(j+d,w),:]
        figure()
        imshow(colored_area)
        glcm = greycomatrix(area,
                        [h*w/30000],
                        [0,np.pi/4,np.pi/2,3*np.pi/4],
                        256,
                        symmetric=True,
                        normed=True)        
        contrast = greycoprops(glcm, 'contrast').mean()
        asm = greycoprops(glcm, 'ASM').mean()
        data.append({'p': p, 
                     'contrast': contrast,
                     'asm': asm})        
        
    ratio = data[0]['contrast']/data[1]['contrast']
    print 'ratio',ratio
    print data
    if (ratio < 0.6):
        p_head = data[1]['p']
    elif (ratio > 3):
        p_head = data[0]['p']
    else:
        if data[0]['asm'] < data[1]['asm']:
            p_head = data[1]['p']
        else:
            p_head = data[0]['p']
        
        

#    areas_prop = sorted(areas_prop)
#    p_tail = areas_prop[0][1]
#    p_head = areas_prop[1][1]
#    
    # Finally, returns head rectangle     
    (i,j) = p_head
    head_area = img_rgb[max(i-d,0):min(i+d,h), max(j-d,0):min(j+d,w), :]
    return head_area

#PATH = 'E:\\Temp\\baleias\\'
#DEST = 'E:\\Temp\\baleias\\heads\\'
#
#for filename in os.listdir(PATH):
#     if os.path.isfile(PATH + filename):
#         print 'Processing', filename
#         head = extract_head(PATH + filename)
#         imsave(DEST + filename, head)
        