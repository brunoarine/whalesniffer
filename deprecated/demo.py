# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:41:36 2015

@author: soldeace
"""

import cv2
from pylab import *
import colorsys
from skimage.filters import threshold_otsu, rank
from skimage.morphology import disk
from skimage import morphology
import numpy as np
from skimage.feature import greycomatrix, greycoprops
from scipy import ndimage as ndi
from simplest_cb import simplest_cb

def fit(img_bw, order):
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
    return img_fit.astype('int')

#PATH = "/home/soldeace/Pictures/whales_subset/"
#FILE = 'w_0.jpg'
PATH = ''
FILE = 'e:\\temp\\baleias\w_170.jpg' # ou sample2.jpg

def imhist(img):
    hist(img.flatten(),256, color = 'black')



#img = imread(PATH + FILE)
img = imread(PATH+FILE)
#img = simplest_cb(img, 0.05)
figure(figsize=(10,8))
subplot(2,2,1)
imshow(img)
title('Original')
for layer_num in range(0,3):
    subplot(2,2,layer_num+2)
    imshow(img[:,:,layer_num], cmap='gray')
    title('RGB layer {}'.format(layer_num))
savefig('baleia_rgb.png')


img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)     
  
figure(figsize=(10,8))
subplot(2,2,1)
imshow(img)
title('Original')
for layer_num in range(0,3):
    subplot(2,2,layer_num+2)
    imshow(img_hsv[:,:,layer_num], cmap='gray')
    title('HSV layer {}'.format(layer_num))
savefig('baleia_hsv.png')


img_s = img_hsv[:,:,1]
img_r = img[:,:,0]
img_g = img[:,:,1]
img_b = img[:,:,2]
img_v = img_hsv[:,:,2]

img_h = img_hsv[:,:,0]
img_h = (img_h.astype('float')/img_hsv.max() * 255).astype('uint8')
img_h[img_h < 51] = 255

global_otsu = threshold_otsu(img_s)
img_bw = img_s < global_otsu
figure()
imshow(img_bw, cmap='gray')
savefig('baleia_bw.png')



(y,x) = np.nonzero(img_bw)
(m,b) = polyfit(x,y,1)
yp = polyval([m,b],x)
figure()
imshow(img_bw, cmap='gray')
plot(x,yp,color='red', scalex=False, scaley=False)
savefig('baleia_linfit.png')




m = len(img_s)
img_blur = cv2.blur(img_s, (m/5,m/5))
blur_otsu = threshold_otsu(img_blur)
img_bw_blur = img_blur < blur_otsu
figure(figsize=(10,5))
subplot(1,2,1)
imshow(img_blur, cmap='gray')
subplot(1,2,2)
imshow(img_bw_blur, cmap='gray')
savefig('baleia_bw_blur.png')




#%% Fit linear da baleia no blur
(y,x) = np.nonzero(img_bw_blur)
table = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
table = table[np.argsort(table[:,0])]
x = table[:,0]
y = table[:,1]
(a,b,c,d) = polyfit(x,y,3)
yp = polyval([a,b,c,d],x)
figure()
imshow(img_bw, cmap='gray')
plot(x,yp,color='red', scalex=False, scaley=False)
#scatter(x,yp,color='red')
#plot([100,200],[300,400], scalex=False, scaley=False)
savefig('baleia_linfit_blur.png')
#%%




kernel = disk(m/40) #np.ones((m/15,m/15),np.uint8)
img_eroded = cv2.erode(img_bw.astype('uint8'), kernel, iterations=1)
img_dilated = cv2.dilate(img_eroded, kernel, iterations=1)
figure(figsize=(15,5))
subplot(1,3,1)
imshow(img_bw, cmap='gray')
title('Original')
subplot(1,3,2)
imshow(img_eroded, cmap='gray')
title('Erodida')
subplot(1,3,3)
imshow(img_dilated, cmap='gray')
title('Dilatada')
savefig('baleia_opened.png')

label_objects, nb_labels = ndi.label(img_dilated)
sizes = np.bincount(label_objects.ravel())
mask_sizes = sizes == sort(sizes)[-2]
img_cleaned = mask_sizes[label_objects]
figure()
imshow(img_cleaned, cmap='gray')
savefig('baleia_cleaned.png')



#%%
(y,x) = np.nonzero(img_cleaned)
table = np.hstack((x.reshape(-1,1),y.reshape(-1,1)))
table = table[np.argsort(table[:,0])]
x = table[:,0]
y = table[:,1]
(a,b,c,d) = polyfit(x,y,3)
yp = polyval([a,b,c,d],x)
figure()
imshow(img_bw, cmap='gray')
plot(x,yp,color='blue', scalex=False, scaley=False)
savefig('baleia_linfit_open.png')
#%%



#skeleton = fit(img_bw_blur, order=3)
skeleton = morphology.skeletonize(img_bw_blur) #fit(img_cleaned, order=3)
figure()
img_overlay = img.copy()
img_overlay[:,:,1][skeleton.astype('bool')] = 255
imshow(img_overlay,cmap='gray')
savefig('baleia_skeleton.png')




(i_idx,j_idx) = np.nonzero(skeleton)
#end_rows = [i_idx.min(),i_idx.max()]
#end_cols = [j_idx.min(),j_idx.max()]
#endpoints = []
#for i in end_rows:
#    for j in end_cols:
#        if skeleton[i,j] == True:
#            endpoints.append((i,j))
#endpoints = np.array(endpoints[:3])
#p1 = endpoints[0]
#p2 = endpoints[1]
p1 = np.nonzero(skeleton[:,j_idx.min()])[0][0], j_idx.min()
p2 = np.nonzero(skeleton[:,j_idx.max()])[0][0], j_idx.max()
p1 = np.array(p1)
p2 = np.array(p2)
endpoints = np.array([p1,p2])

delta = int(np.linalg.norm(p2-p1)/6)
img_area = img.copy()
n=1
for p in endpoints:    
    (i,j) = p    
    cv2.rectangle(img_area,(j-delta,i-delta),(j+delta,i+delta),(0,255,0),2)
    #cv2.putText(img_area,n,p,'FONT_HERSHEY_SIMPLEX', 12, (0,255,0))
    n = n+1
figure()
imshow(img_area)
scatter(endpoints[:,1], endpoints[:,0], color='red')
#annotate('p1',p1)
#annotate('p2',p2)
savefig('baleia_endpoints.png')




figure(figsize=(10,5))
counter = 1
for p in endpoints:
    used_img = img_h #img_h
    (i,j) = p
    (m,n) = used_img.shape
    area = used_img[max(i-delta,0):min(i+delta,m), max(j-delta,0):min(j+delta,n)]
    glcm = greycomatrix(area,
                        [m*n/30000],
                        [0,pi/4,pi/2,3*pi/4],
                        256,
                        symmetric=True,
                        normed=True)
    feature = 'contrast'
    prop = greycoprops(glcm, feature).mean()
    subplot(1,2,counter)
    imshow(area, cmap='gray')
    title('P{} ({} = {})'.format(counter,feature,prop))    
    counter = counter + 1
    


savefig('baleia_glcm.png')
    