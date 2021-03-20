# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 08:28:28 2015

@author: soldeace
"""

import numpy as np
from matplotlib.pyplot import imshow, plot

def parabola(img, centro, p_min, p_max):
    (m,n) = img.shape
    centro_x = centro[0]
    centro_y = centro[1]
    vector_p = np.linspace(-p_max,p_max,50);
    vector_phi=np.linspace(0,2*np.pi-(2*np.pi/100),50);
    Accumulator = np.zeros((len(vector_phi),len(vector_p)));
    (y,x) = np.nonzero(img)
    
    for i in range(len(x)):
       for j in range(len(vector_phi)):
           Y=y[i]-centro_y;
           X=x[i]-centro_x;
           angulo=vector_phi[j];
           numerador=(Y*np.cos(angulo)-X*np.sin(angulo))**2;
           denominador=4*(X*np.cos(angulo)+Y*np.sin(angulo));
           if denominador != 0:
               p=numerador/denominador;           
               if (abs(p) > p_min) & (abs(p) < p_max) & (p != 0):
                   indice=np.nonzero(vector_p>=p)[0][0]               
                   Accumulator[j,indice] = Accumulator[j,indice]+1;
               
    maximo=Accumulator.max()
    (idx_phi,idx_p)= np.nonzero(Accumulator==maximo);
    p=vector_p[idx_p][0];
    phi=vector_phi[idx_phi][0];
    imshow(Accumulator, cmap='afmhot')
    return (p,phi)