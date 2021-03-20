# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:29:09 2015

@author: Usuario
"""

from matplotlib.image import imread
from matplotlib.pyplot import imshow, scatter, plot, colorbar
from skimage.color import rgb2hsv
from skimage.filters import rank
from skimage.filters import sobel
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_adaptive
from simplest_cb import simplest_cb
from scipy.signal import convolve2d
import numpy as np
import operator
from skimage.morphology import binary_opening, binary_closing, disk
from skimage.morphology import skeletonize, binary_erosion, dilation, reconstruction
from scipy import ndimage as ndi
from scipy import fftpack
from skimage.feature import local_binary_pattern
import hough
from skimage.feature import greycomatrix, greycoprops
from skimage.transform import rotate
from skimage import exposure
from itertools import permutations
from skimage.measure import moments_hu
from skimage.feature import match_template
from pylab import figure, title
import cv2
from skimage import transform as tf
from skimage.io import imsave
import pandas as pd
import os

file = 'e:\\temp\\baleias\\w_115.jpg'


#################################################################
def show(img, cmap='cubehex'):
    imshow(img, cmap=cmap)
    colorbar();


def angle_between(p1, p2):
    angle = np.arctan(float(p1[0]-p2[0])/float(p1[1]-p2[1]))
    return np.rad2deg(angle % (2 * np.pi))


def rotate_coord(p, theta):
    theta = theta*2*np.pi/360.    
    y = float(p[0])
    x = float(p[1])
    axis_y = h/2.
    axis_x = w/2.
    X = x - axis_x
    Y = y - axis_y
    new_x = X*np.cos(theta) - Y*np.sin(theta)
    new_y = Y*np.sin(theta) - Y*np.cos(theta)
    return np.array([new_y+axis_y, new_x+axis_x]).astype('int')
    


def find_endpoints(img):
    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(img)    
    # Initialize empty list of co-ordinates
    skel_coords = []    
    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):
        # Extract an 8-connected neighbourhood
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')
        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = img[row_neigh,col_neigh].ravel() != 0
        # If the number of non-zero locations equals 2, add this to 
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r,c))
    furthest_dist = 0
    furthest_points = []
    for points in permutations(skel_coords, 2):
        p1 = np.array(points[0])
        p2 = np.array(points[1])
        distance = np.linalg.norm(p2-p1)
        if distance > furthest_dist:
            furthest_points = [p1,p2]
            furthest_dist = distance            
    return np.array(furthest_points)



def kullback_leibler_divergence(p, q):
    p = np.asarray(p)
    q = np.asarray(q)
    filt = np.logical_and(p != 0, q != 0)
    return np.sum(p[filt] * np.log2(p[filt] / q[filt]))


def estimate_noise(I, mode='sobel'):  
    (h,w) = I.shape
    if mode == 'sobel':        
        sigma = np.sum(sobel(I))/float(h*w)        
    elif mode == 'fft':
        F1 = fftpack.fft2(I)
        F2 = fftpack.fftshift(F1)
        psd2D = np.abs( F2 )**2
        logimg = np.log(psd2D)
        figure()        
        show(logimg, cmap='inferno')
        sigma = np.std(logimg[h/2-20:h/2+20, w/2-20:w/2+20])
    return sigma

def denoise(img, mode=rank.median):    
    (h,w) = img_r.shape
    blur_shape = (int(h/20.), int((h/20.)))
    output = mode(img, np.ones(blur_shape))
    return output
    
    
def fit(img_bw, mode='poly', order=3):
    if mode == 'skeletonize':
        img_fit = skeletonize(img_bw)
    elif mode == 'poly':
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







def get_regions(file, autowb=False, wb_cutoff=0.01, sel_mode='sobel',
                skel_mode='skeletonize'):

    img_rgb = imread(file)
    
    #############################################################
    # EQUALIZING EXPOSURE
    #############################################################
    if autowb == True:    
        img_rgb = simplest_cb(img_rgb, wb_cutoff)
    
    
    #############################################################  
    # DEFINING CHANNELS
    #############################################################       
    img_hsv = rgb2hsv(img_rgb)
    img_r = img_rgb[:,:,0]
    img_s = img_hsv[:,:,1]
    img_h = img_hsv[:,:,0]
    img_h[img_h < 0.4] = 1   
    
    h,w = img_r.shape
    #############################################################
    # Which channel is the smoothest?
    #############################################################  
    
    candidates = [img_h, img_s, img_r] 
    noises = []
    for idx, channel in enumerate(candidates):    
        noise_value = estimate_noise(channel, mode=sel_mode)    
        noises.append((idx, noise_value))
    img_ref = candidates[min(noises, key=operator.itemgetter(1))[0]]
    
    #############################################################
    # Remove noise    
    #############################################################
    if min(noises, key=operator.itemgetter(1))[0] == 2:
        img_smoothed = rank.median(img_ref, disk(8))
        segmented = canny(img_smoothed, sigma=0.8)
    else:   
        img_smoothed = denoise(img_ref, mode=rank.median)
        segmented = rank.gradient(img_smoothed, disk(h*w/55756))    
   
    #############################################################
    # THRESHOLDING
    #############################################################
    threshold = threshold_otsu(segmented)
    binary = segmented > threshold   
    
    #############################################################
    #Closing
    #############################################################
    closed = binary_closing(binary, disk(20))
    
    #############################################################
    # SMALLER BLOBS REMOVAL
    #############################################################
    label_objects, nb_labels = ndi.label(closed)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes == np.sort(sizes)[-2]
    img_cleaned = mask_sizes[label_objects]
    
    ############################################################
    # FILLING HOLES AND CLOSING GAPS
    ############################################################
    img_closedgaps = dilation(img_cleaned, disk(30))    
    img_fill = ndi.binary_fill_holes(img_closedgaps)
    
       
    ############################################################
    # FITTING
    ############################################################
    skeleton = fit(img_fill, mode=skel_mode)
    
      
    ############################################################
    # FINDING ENDPOINTS TO THE FIT
    ############################################################
    endpoints = find_endpoints(skeleton)
    

    
    
    ############################################################
    # DETERMINING QUADRANTES
    ############################################################
    d = w/10
    
    evaluation_img = binary_erosion(img_fill, disk(15))
    
    values = []
    
    for point_num, p in enumerate(endpoints):    
        (i,j) = p
        interest = np.nonzero(img_fill)
        imin, imax = interest[0].min(), interest[0].max()
        jmin, jmax = interest[1].min(), interest[1].max()
        area = evaluation_img[max(i-d,imin):min(i+d,imax), max(j-d,jmin):min(j+d,jmax)]
        area_color = img_r[max(i-d,imin):min(i+d,imax), max(j-d,jmin):min(j+d,jmax)]
        
        #area_color = (255*(img_r - np.min(img_r))/(np.max(img_r) - np.min(img_r))).astype('uint8')
        
        (m,n) = area.shape    
        lbp = local_binary_pattern(area, 4, 18, 'uniform')
        n_bins = lbp.max() + 1
        hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))         
        
        contour = canny(area)        
        contour = binary_closing(contour, disk(3))
        contour = ndi.binary_fill_holes(contour)
        mom = cv2.moments(contour.astype('uint8'))
        hu_mom = cv2.HuMoments(mom)
        
        hu_distance = np.linalg.norm(hu_mom)
        glcm = greycomatrix(area_color,
                            [m*n/30000, m*n/15000, m*n/10000],
                            [0,np.pi/4,np.pi/2,3*np.pi/4],
                            256,                        
                            normed=True)
        features = ['contrast', 'correlation', 'energy', 'homogeneity']
        texture = [greycoprops(glcm, feature).mean() for feature in features]
        
        filename = file.split('\\')[-1]
        area_name = 'e:\\temp\\baleias\\areas\\{}{}'.format(point_num, filename)        
        imsave(area_name, area_color)
               
        values.append(np.concatenate(([area_name.split('\\')[-1]],
                                      hist,
                                      hu_mom[:,0],
                                      texture)))
        print 'lbp', hist
        print 'hu', hu_mom
        print 'texture', texture
    return values
        

path = 'e:\\temp\\baleias\\'

table = []
def get_from_files():
    PATH = 'E:\\Temp\\baleias\\'    
    for filename in os.listdir(path):
        if os.path.isfile(path + filename):             
            print 'Processando', filename            
            new_lines = get_regions(path + filename, autowb=True)             
            table.append(new_lines[0])
            table.append(new_lines[1])

    np.savetxt('params_table.txt',table)
    


header = ['filename', 'l1', 'l2', 'l3', 'l4', 'l5', 'h1', 'h2', 'h3', 'h4',
          'h5', 'h6', 'h7', 'contrast', 'correlation', 'energy', 'homogeneity',
          "unknown"]
df = pd.DataFrame(table, columns=header)
df = df.sort_values(by='filename')
df.to_csv('params.csv')