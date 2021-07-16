# -*- coding: utf-8 -*-

from pylab import imread
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab
from skimage.filters import gaussian
from sklearn.cluster import KMeans
import mahotas
from scipy import ndimage as ndi
from ..utils import blob


class Clustering:

    def predict(self, filenames_list, scale=0.4, n_clusters=4):            
        if not isinstance(filenames_list, list):
            raise Exception('Input list of files is not a list actually')
        rectangles = []
        for filename in filenames_list:
            img_orig = gaussian(rgb2lab(imread(filename))[:,:,1], 3)
            h_orig, w_orig = img_orig.shape
            h_small, w_small = int(h_orig * scale), int(w_orig * scale)
            thumb = resize(img_orig, (h_small, w_small))
            
            
            model = KMeans(n_clusters)
            segments_flatten = model.fit_predict(thumb.reshape(-1,1))
            segments = segments_flatten.reshape(h_small,w_small)        
            seg_means = [np.mean(thumb[segments == n]) for n in range(np.max(segments)+1)]
            brightest_seg = np.argmax(seg_means)
            mask = segments == brightest_seg
            thumb_closed = mahotas.close(mask, np.ones((30*scale,30*scale)))        
            
            label_objects, nb_labels = ndi.label(thumb_closed)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes == np.sort(sizes)[-2]
            img_cleaned = mask_sizes[label_objects]
            
            img_final = resize(img_cleaned, (h_orig, w_orig)).astype(int)
            rectangles.append(blob.bound_rect(img_final))              
        return rectangles
            
#x = color.normalize_RGBratio(imread('/home/soldeace/Pictures/small/w_17.png'))[:,:,0]



#aproximatepdf
# 1/sqrt(2*np.pi()) * np.exp(-1/2*x**2)
#
#def wrapper(func, *args, **kwargs):
#    def wrapped():
#         return func(*args, **kwargs)
#    return wrapped
#
#lognormdemora = lambda x: np.log(norm.pdf(x))
#
#wrap = wrapper(lognormd, x)
#print timeit.timeit(wrap, number=10000)
#
#def optipower(b, p):
#    """
#    Calculates b^p
#    Complexity O(log p)
#    b -> double
#    p -> integer
#    res -> double
#    """
#    res = 1
#
#    while p:
#        if p & 0x1: res *= b
#        b *= b
#        p >>= 1
#
#    return res