# -*- coding: utf-8 -*-

import numpy as np
from skimage.color import rgb2lab
from skimage.filters import gaussian
from skimage.morphology import disk
from skimage.io import imread
import mahotas
from scipy import ndimage as ndi
from ..utils import blob
from ..utils import color


class Manual:
    """
    Try to segmentate whales by identifying the largest and brightest blob in the image (after a gaussian blur)
    """

    def predict(self, filenames_list, threshold=0.75):        
        if not isinstance(filenames_list, list):
            raise Exception('Input list of files is not a list actually')
        
        rectangles = []
        for filename in filenames_list:
            img = imread(filename)
            img_a = color.scaler(gaussian(rgb2lab(img)[:,:,1], 3))
            h,w = img_a.shape
            mask = img_a > threshold
            img_closed = mahotas.close(mask, disk(h/15))        
            
            label_objects, nb_labels = ndi.label(img_closed)
            sizes = np.bincount(label_objects.ravel())
            mask_sizes = sizes == np.sort(sizes)[-2]
            img_cleaned = mask_sizes[label_objects]
            rectangles.append(blob.bound_rect(img_cleaned))
        return rectangles