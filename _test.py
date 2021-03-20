# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:12:52 2015

@author: soldeace
"""

import sklearn.mixture
import numpy as np
from pylab import imread, imshow,figure
from skimage.color import rgb2lab
import matplotlib.pyplot as plt
from whalehunt.utils import color
from skimage.filters import rank, threshold_otsu, gaussian_filter


def imhist(img, range=None):
    plt.hist(img.flatten(),256, color = 'black', range=range)

def solve(means,stds):
  m1 = means[0]
  m2 = means[1]
  std1 = stds[0]
  std2 = stds[1]
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])



img = imread('samples/sample1.jpg')
img_a = gaussian_filter(color.scaler(rgb2lab(img)[:,:,1]),10)
(h,w) = img_a.shape
figure()    
imhist(img_a)
figure()
imshow(img_a, cmap='gray')
X = img_a.reshape(-1,1)    
model = sklearn.mixture.GMM(4)
model.fit(X)

sds = np.sqrt(model.covars_).ravel()
means = model.means_.ravel()
p = np.argsort(means)
intersect = min(solve(means[p][-2:], sds[p][-2:]))

