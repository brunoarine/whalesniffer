# -*- coding: utf-8 -*-

from pylab import imread, imshow, figure, colorbar
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2lab
from skimage.filters import gaussian
import mahotas
from ..utils import blob
from scipy import ndimage as ndi
from skimage.morphology import disk


def show(img, cmap='jet'):
    figure(figsize=(7.5,5))    
    imshow(img, cmap=cmap)
    colorbar();


def normpdf(x, mean=0.0, sd=1.0):
    var = sd**2    
    c = 2.5066282746310002
    num = 1/(sd*c) * np.exp(-(x-mean)**2/(2*var))
    return num

def lognormpdf(x,u=0,s=1):
    a = -np.log(s*2.5066282746310002)
    b = -(x - float(u))**2/(2*float(s + u*0.025)**2)
    return a+b

def least_dist(i, idx_array, num_cols, method='manhattan'):
    ''' Returns the Manhattan distance of the nearest point in idx_array
    from i. Why Manhattan? Because Euclidean distances are quite more expensive
    to compute.

    Parameters
    ----------
    
    i : int
        Index of reference point to which you want to compute distance.
    idx_array : array or list
        Set of available points
    num_cols : int
        Since we are working with a flattened image at this point, the number
        of columns of the image must be provided for the distances to be
        calculated.
        
    '''   
    # Making sure idx_array is an array, for % and / doesn't work with lists
    idx_array = np.array(idx_array)
    # The function below with % and / works pretty much the same as
    # MATLAB's ind2sub
    y = abs(i%num_cols - idx_array%num_cols)
    x = abs(i/num_cols - idx_array/num_cols)
    if method == 'manhattan':
        dist_array = y + x                    
    if method == 'euclidean':
        dist_array = np.sqrt(y**2 + x*2)
    return np.min(dist_array)
    
    

class Segment:        
    def __init__(self, prior=None, idx=None):
        if prior is None:
            self.priors_ = []
            self.idx_ = []
        else:
            self.priors_ = [prior]
            self.idx_ = [idx]
    def evidence(self, posteriors, idx):
        self.priors_.append(posteriors)
        self.idx_.append(idx)
    def prob(self, x, i, num_cols):        
        u_priors = np.mean(self.priors_)
        s_priors = np.std(self.priors_)
        print(s_priors)
        u_dist = 1
        s_dist = num_cols/25
        log_p_a = lognormpdf(x, u_priors, s_priors)
        log_p_dist = lognormpdf(least_dist(i, self.idx_, num_cols, 'euclidean'), u_dist, s_dist)
        return  log_p_a + log_p_dist
#        
        

class Bayesian:
    segments_ = [Segment()]
    
    def __init__(self, scale=0.05, cutoff=-10):
        self.scale = scale
        self.cutoff = cutoff
    
    def predict(self, filenames_list):
        if not isinstance(filenames_list, list):
            raise Exception('Input list of files is not a list actually')
        
        rectangles = []
        for filename in filenames_list:
            img_orig = gaussian(rgb2lab(imread(filename))[:,:,1], 3)
            show(img_orig)
            h_orig, w_orig = img_orig.shape
            h_small, w_small = int(h_orig * self.scale), int(w_orig * self.scale)
            thumb = resize(img_orig, (h_small, w_small)).ravel()
            
            m = len(thumb)
            pixel_order = range(m)
            res = np.zeros(len(thumb)).astype('int')
            #np.random.shuffle(pixel_order)
            
            n = 0
            self.segments_[0].evidence(thumb[0],0)
            for i in pixel_order:
                n += 1
                likelihoods = np.array([segment.prob(thumb[i],i, w_small) for segment in self.segments_])
                priors = np.log([len(segment.idx_)/float(n+1) for segment in self.segments_])            
                if ((likelihoods + priors) < self.cutoff).all(): 
                    self.segments_.append(Segment(thumb[i],i))
                    res[i] = len(self.segments_) - 1
                else:
                    
                    self.segments_[np.argmax(likelihoods)].evidence(thumb[i],i)                 
                    res[i] = np.argmax(likelihoods)

                           
            show(res.reshape(h_small,w_small))
            seg_means = [np.mean(seg.priors_) for seg in self.segments_]
            brightest_seg = np.argmax(seg_means)
            print(brightest_seg)
            print(res.reshape(h_small, w_small))
            mask = np.zeros(len(res)).astype('int')
            mask[res == brightest_seg] = 1           
            
            img_mask = mask.reshape(h_small, w_small)
            print(seg_means)
            show(img_mask)
            img_closed = mahotas.close(img_mask, disk(h_orig/13))
            label_objects, nb_labels = ndi.label(img_closed)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes == np.sort(sizes)[-2]
            img_cleaned = mask_sizes[label_objects]
            
            figure()
            imshow(img_cleaned)            
            
            rectangles.append(blob.bound_rect(img_cleaned))
        return rectangles
