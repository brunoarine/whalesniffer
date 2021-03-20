# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 14:29:09 2015

@author: Usuario
"""
import matplotlib
from matplotlib.image import imread
from matplotlib.pyplot import imshow, scatter, plot, colorbar
from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import rank
from skimage.filters import sobel
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_adaptive, gaussian_filter
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
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.facecolor'] = '1.0'

file = '../samples/sample4.jpg'


#################################################################
def show(img, cmap='gray'):
    figure(figsize=(7.5,5))    
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
        figure()
        show(sobel(I), cmap='gray')
        sigma = np.sum(sobel(I))/float(h*w)        
        print sigma
    elif mode == 'fft':
        F1 = fftpack.fft2(I)
        F2 = fftpack.fftshift(F1)
        psd2D = np.abs( F2 )**2
        logimg = np.log(psd2D)
        figure()        
        imshow(logimg, cmap='inferno')
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



def imhist(img, range=None):
    plt.hist(img.flatten(),256, color = 'black', range=(0,1), normed=True)


img_rgb = imread(file)
#############################################################
# EQUALIZING EXPOSURE
#############################################################
#img_rgb = simplest_cb(img_rgb, 0.001)#exposure.equalize_adapthist(img_rgb)







#############################################################  
# DEFINING CHANNELS
#############################################################  


#img_rgb = simplest_cb(img_rgb, 0.05)
img_hsv = rgb2hsv(img_rgb)
img_r = img_rgb[:,:,0]
img_g = img_rgb[:,:,1]
img_b = img_rgb[:,:,2]
img_v = img_hsv[:,:,2]
img_s = img_hsv[:,:,1]
img_h = img_hsv[:,:,0]
img_h[img_h < 0.4] = 1
img_lab = rgb2lab(img_rgb)
img_a = img_lab[:,:,1]

figure()
imshow(img_rgb)


figure()
imshow(img_s, cmap='gray')


figure()
imshow(img_h, cmap='gray')



h,w = img_r.shape
#############################################################
# Which channel is the smoothest?
#############################################################  

candidates = [img_h, img_s, img_r] 
noises = []
for idx, channel in enumerate(candidates):    
    noise_value = estimate_noise(channel, mode='sobel')    
    noises.append((idx, noise_value))
img_ref = candidates[min(noises, key=operator.itemgetter(1))[0]]
#img_ref = img_h

figure()
imshow(img_ref, cmap='gray')

#############################################################
# Remove noise    
#############################################################
if min(noises, key=operator.itemgetter(1))[0] == 2:
    img_smoothed = rank.median(img_ref, disk(8))
    segmented = canny(img_smoothed, sigma=0.8)
else:   
    img_smoothed = denoise(img_ref, mode=rank.median)
    segmented = rank.gradient(img_smoothed, disk(h*w/55756))

figure()
imshow(img_smoothed, cmap='gray')


figure()
imshow(segmented, cmap='gray')


#############################################################
# THRESHOLDING
#############################################################
threshold = threshold_otsu(segmented)
binary = segmented > threshold

figure()
imshow(binary, cmap='gray')


#############################################################
#Closing
###################
closed = binary_closing(binary, disk(20))

figure()
imshow(closed, cmap='gray')

#############################################################
# SMALLER BLOBS REMOVAL
#############################################################
label_objects, nb_labels = ndi.label(closed)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes == np.sort(sizes)[-2]
img_cleaned = mask_sizes[label_objects]

figure()
imshow(img_cleaned, cmap='gray')
title('img_cleaned')
############################################################
# FILLING HOLES AND CLOSING GAPS
############################################################
img_closedgaps = dilation(img_cleaned, disk(30))

figure()
imshow(img_closedgaps, cmap='gray')
title('img_closedgaps')


img_fill = ndi.binary_fill_holes(img_closedgaps)

figure()
imshow(img_fill, cmap='gray')
title('img_fill')



############################################################
# FITTING
############################################################
skeleton = fit(img_fill, mode='skeletonize')


figure()
imshow(skeleton, cmap='gray')


############################################################
# FINDING ENDPOINTS TO THE FIT
############################################################

#(i_idx,j_idx) = np.nonzero(skeleton)
#p1 = np.nonzero(skeleton[:,j_idx.min()])[0][0], j_idx.min()
#p2 = np.nonzero(skeleton[:,j_idx.max()])[0][0], j_idx.max()
#p1 = np.array(p1)
#p2 = np.array(p2)
#endpoints = np.array([p1,p2])
endpoints = find_endpoints(skeleton)

figure()
imshow(img_rgb)
scatter(endpoints[:,1], endpoints[:,0], color='red')


############################################################
# ROTATION
############################################################
angle = angle_between(endpoints[0], endpoints[1])



############################################################
# DETERMINING QUADRANTES
############################################################
d = w/10
regions = []
evaluation_img = binary_erosion(img_fill, disk(15))

for point_num, p in enumerate(endpoints):    
    (i,j) = p
    interest = np.nonzero(img_fill)
    imin, imax = interest[0].min(), interest[0].max()
    jmin, jmax = interest[1].min(), interest[1].max()
    area = evaluation_img[max(i-d,imin):min(i+d,imax), max(j-d,jmin):min(j+d,jmax)]
    area_color = img_r[max(i-d,imin):min(i+d,imax), max(j-d,jmin):min(j+d,jmax)]
    (m,n) = area.shape    
#    area_smooth = rank.median(area, np.ones((15,15)))
#    area_threshold = area_smooth > threshold_otsu(area_smooth)    
#    area_color = img_rgb[max(i-d,0):min(i+d,h), max(j-d,0):min(j+d,w),:]      
#    area_edges = canny(area_threshold)
#    radius, angle = hough.parabola(area_edges, (n/2, m/2), d, d*2)
#    angle_degree = (2*np.pi - angle + np.pi)*180/np.pi
#    area_rotated = rotate(area, angle_degree)
    lbp = local_binary_pattern(area, 6, 18, 'uniform')
    n_bins = lbp.max() + 1
    hist, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins)) 
    figure()
    plot(hist)
    figure()
    #score = kullback_leibler_divergence(hist, refhist)
    contour = canny(area)
    #contour = (contour > threshold_otsu(contour)).astype('int')
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
    regions.append((point_num,
                    p,                    
                    area_color,
                    np.sum(np.log(hist[hist != 0])),
                    hu_distance,
                    texture[0],
                    texture[1],
                    texture[2],
                    texture[3]))
    
    figure()
    imshow(area_color, cmap='gray')
    title('P{} hu distance = {}'.format(point_num,hu_distance))   
    


print 'P0 > P1:', np.array(regions[0][3:]) > np.array(regions[1][3:])
############################################################
# CLASSIFYING QUADRRANTES
############################################################
    
#head = min(regions, key=operator.itemgetter(4))[3]
#imshow(head, cmap='gray')
    



############################################################
# SAVING
############################################################
def smoothness(img):
    """calculate the entropy of an image"""
    histogram = exposure.histogram(img)[0]
    histogram_length = histogram.sum()
    samples_probability = [float(h) / histogram_length for h in histogram]
    entropy = -sum([p * math.log(p, 2) for p in samples_probability if p != 0])
    energy = sum([p**2 for p in samples_probability])
    u2 = moment(histogram, 2)
    u4 = moment(histogram, 4)
    return (1/u2 + 1/u4 + energy)*entropy
    
    
def crop(img):
    interest = np.nonzero(img)
    imin, imax = interest[0].min(), interest[0].max()
    jmin, jmax = interest[1].min(), interest[1].max()
    output = img[imin:imax, jmin:jmax]
    return output
    
imhist(img)
plt.plot(mlab.normpdf(x,0.299,0.05), c='red')
plt.show()
