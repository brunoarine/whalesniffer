# -*- coding: utf-8 -*-

from skimage.color import rgb2hsv, rgb2lab
from skimage.filters import rank, threshold_otsu, gaussian
import numpy as np
from skimage.morphology import disk
import mahotas
from scipy import ndimage as ndi
from skimage.io import imread
from ..utils import color, noise, gauss, blob


class Thresholding():
    '''
    Detects whale through thresholding.
    
    Parameters
    ----------
    
    denoise : function, optional
        Used function to clear up the image before segmentation.
        Default is skimage.filters.rank.median
    '''
    
    def __init__(self,
                 denoise_mode=rank.median,
                 candidates=['h','a'],
                 combine=False,
                 threshold='gauss'):        
        self.denoise_mode = denoise_mode
        self.candidates = candidates
        self.combine = combine
        self.threshold = threshold
        
    # for compatibility sake
    def fit(self,*args):
        return self 
    
    def predict(self, filenames_list):
        if not isinstance(filenames_list, list):
            raise Exception('Input list of files is not a list actually')
        
        rectangles = []
        for filename in filenames_list:
            img_rgb = imread(filename)
            img_hsv = rgb2hsv(img_rgb)
            img_nrgb = color.normalize_RGBratio(img_rgb)
            img_lab = rgb2lab(img_rgb)
            
            img_h = img_hsv[:,:,0]
            img_h[img_h < 0.4] = 1 - img_h[img_h < 0.4]
            
            # TODO add normalized RGB and opposite RGB            
            channel = {'r': img_rgb[:,:,0],
                       'g': img_rgb[:,:,1],
                       'b': img_rgb[:,:,2],
                       'h': img_hsv[:,:,0],
                       's': img_hsv[:,:,1],
                       'v': img_hsv[:,:,2],
                       'nr': img_nrgb[:,:,0],
                       'ng': img_nrgb[:,:,1],
                       'nb': img_nrgb[:,:,2],
                       'l': img_lab[:,:,0],
                       'a': img_lab[:,:,1],
                       'b': img_lab[:,:,2]                        
                       }
            
            ### Selects best channel
            if self.combine:
                chan = 1.
                for x in self.candidates:
                    chan *= color.scaler(channel[x])
            else:           
                chan = noise.least_noise([channel[x] for x in self.candidates])          
            (h,w) = chan.shape
            
            chan = color.scaler(chan)
            ### Executes further denoising, which helps later on
            #chan = noise.denoise(chan, mode=self.denoise_mode)
            chan = gaussian(chan, 5)
            
            
            ### Finding binary edges in the smoothed image            
            #chan = rank.gradient(chan, disk(int(h*w/55756.)))              
            chan = color.scaler(chan)
            if self.threshold == 'otsu':
                chan = chan > threshold_otsu(chan)
            elif self.threshold == 'gauss':
                n_dist = 3 if channel['s'].std() > 0.05 else 4
                chan = chan > gauss.threshold_gauss(chan, n_dist)
            elif self.threshold == 'kmeans':
                n_dist = 3 if channel['s'].std() > 0.05 else 4
                chan = chan > gauss.threshold_kmeans(chan, n_dist)
            
                        
            ### Transforming contours into shapes at last by closing gaps
            # For sample3.jpg, these are the total time for each function:            
            # skimage's dilation: 11 s
            # scipy's dilation: 7 s
            # mahota's dilation: 3 s
            chan = mahotas.dilate(chan, disk(h/46))
            chan = mahotas.erode(chan, disk(h/46))             
            
            chan = ndi.binary_fill_holes(chan)
                        
            ### Selects largest contour, supposedly to be the whale            
            label_objects, nb_labels = ndi.label(chan)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes == np.sort(sizes)[-2]
            chan = mask_sizes[label_objects]
            
            ### Draw boundary rectangle            
            rectangles.append(blob.bound_rect(chan))
                        
        return rectangles
