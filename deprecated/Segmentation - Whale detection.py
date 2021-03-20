# -*- coding: utf-8 -*-

import numpy as np
from skimage import io
from skimage.filters.rank import *
from skimage.color import rgb2grey
import skimage
from sklearn.cluster import AgglomerativeClustering

    
def import_img(fName):
    try:
        res = io.imread(fName + '.jpg')
    except:
        res = io.imread(fName)
    return res

def normalize_RGBratio(image):
    res = image.copy()
    mean = image.mean()
    res[:,:,1] = image[:,:,1]/image[:,:,0]*mean
    res[:,:,2] = image[:,:,2]/image[:,:,0]*mean
    res[:,:,0] = mean
    return res

def resize_seg(seg,image):
    
    row = np.sum(seg, axis=1)
    col = np.sum(seg, axis=0)
    
    seg_dim = len(row)
    
    (a,b,c) = image.shape

    for i in range(seg_dim):
        j = 99-i
        if row[i]>0: 
            row_max=min(a, int(i*a/seg_dim) + int(5*a/seg_dim))
        if col[i]>0: 
            col_max=min(b, int(i*b/seg_dim) + int(5*b/seg_dim))
        if row[j]>0: 
            row_min=max(0, int(j*a/seg_dim) - int(5*a/seg_dim))
        if col[j]>0: 
            col_min=max(0, int(j*b/seg_dim) - int(5*b/seg_dim))        
    res = np.zeros((a,b),dtype=np.int8)
    res[row_min: row_max+1, col_min: col_max+1] = int(1)    
        
    return res

def seg_stats_retrieve(seg,image):
    data = []
    for seg_no in range(seg.max()+1):
        data_point = np.mean(image[seg==seg_no],axis=0)
        data.append(data_point)
    return np.vstack(data)

def seg_reduction(seg, data, new_seg_no=3):   
    clus = AgglomerativeClustering(n_clusters=new_seg_no, affinity='euclidean')
    new_clust = np.array(clus.fit_predict(data),dtype=np.int8)  
    new_seg = seg.copy()
    for seg_no in range(seg.max()+1):
        new_seg[seg==seg_no] = new_clust[seg_no]    
    return new_seg

def seg_selection(seg, image_copy, image_norm):
    ratio_min= float('inf')
    seg_whale_no = 0
    for seg_no in range(seg.max()+1):
        values = np.mean(image_norm[seg==seg_no], axis=0)
        ratio = np.mean(abs(values/values[0]-1))
        img_grey = rgb2grey(image_copy)
        ratio_light = np.mean(img_grey[seg==seg_no])/np.mean(img_grey)
        
        if ratio*ratio_light < ratio_min:
            seg_whale_no = seg_no
            ratio_min=ratio*ratio_light
            
    res = 0 + (seg==seg_whale_no)
    return res

def whale_detection(image):
    img_copy = skimage.transform.resize(image,(100,100,3))
    img_copy = skimage.filters.gaussian_filter(img_copy, sigma=0.5)
    img_norm = normalize_RGBratio(img_copy)
    seg = skimage.segmentation.slic(img_norm, n_segments=10,max_iter=100,enforce_connectivity=True,
                                   max_size_factor=10)
    
    data = seg_stats_retrieve(seg, img_copy)
    seg = seg_reduction(seg, data)
    seg = seg_selection(seg, img_copy, img_norm)
    res = resize_seg(seg,image)
    
    crop = np.nonzero(res)    
    rectangle = np.array([[crop[0].min(), crop[1].min()],
                          [crop[0].max(), crop[1].max()]])
    return rectangle



