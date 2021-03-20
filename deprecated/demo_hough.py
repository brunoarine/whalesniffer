# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:28:28 2015

@author: soldeace
"""

from pylab import *
from skimage.filters import canny

img = imread('parabola.png')
img = img[:,::-1,0]
img = (img > 0.5).astype('int')
imshow(img, cmap='gray')

def hough_parabola(img, centro, p_min, p_max):
    (m,n) = img.shape
    centro_x = centro[0]
    centro_y = centro[1]
    vector_p = linspace(-pmax,pmax,50);
    vector_phi=linspace(0,2*pi-(2*pi/100),50);
    Accumulator = zeros((len(vector_phi),len(vector_p)));
    (y,x) = np.nonzero(img)
    
    for i in range(len(x)):
       for j in range(len(vector_phi)):
           Y=y[i]-centro_y;
           X=x[i]-centro_x;
           angulo=vector_phi[j];
           numerador=(Y*cos(angulo)-X*sin(angulo))**2;
           denominador=4*(X*cos(angulo)+Y*sin(angulo));
           if denominador != 0:
               p=numerador/denominador;           
               if (abs(p) > pmin) & (abs(p) < pmax) & (p != 0):
                   indice=np.nonzero(vector_p>=p)[0][0]               
                   Accumulator[j,indice] = Accumulator[j,indice]+1;
               
    maximo=Accumulator.max()
    (idx_phi,idx_p)= np.nonzero(Accumulator==maximo);
    p=vector_p[idx_p][0];
    phi=vector_phi[idx_phi][0];
    return (p,phi)