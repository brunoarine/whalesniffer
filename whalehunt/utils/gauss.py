# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 15:47:34 2015

@author: soldeace
"""

import numpy as np
import sklearn.mixture
from skimage.transform import resize
import sklearn.cluster

def solve(means,stds):
  m1 = means[0]
  m2 = means[1]
  std1 = stds[0]
  std2 = stds[1]
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])

def threshold_gauss(image, n_dists=4):
    img = resize(image,(30,30))
    (h,w) = img.shape
    X = img.reshape(-1,1)    
    model = sklearn.mixture.GMM(n_dists)
    model.fit(X)    
    means = model.means_.ravel()
    p = means.argsort()
    means = means[p][-2:]
#    weights = weights[p][-2:]
#    solution = means*weights/weights.sum()
    return means.sum() * 0.5
    
    
def threshold_kmeans(image, n_dists=4):
    img = resize(image,(30,30))
    (h,w) = img.shape
    X = img.reshape(-1,1)    
    model = sklearn.cluster.KMeans(n_dists)
    model.fit(X)    
    means = model.cluster_centers_.ravel()
    p = means.argsort()
    means = means[p][-2:]
#    weights = weights[p][-2:]
#    solution = means*weights/weights.sum()
    return means.sum() * 0.5