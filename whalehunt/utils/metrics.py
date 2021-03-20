# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:22:37 2015

@author: soldeace
"""

def evaluate_score(a, b, score):
    '''
    Computes a score based on the intersection between two bounding rectangles.
    
    Parameters
    ----------
    a : 2-d point
        Actual region in the image
    b : 2-d point
        Base, the reference base you wish to compare with the actual region
    
    '''
    xa0 = a[0,1]
    xb0 = b[0,1]
    xa1 = a[1,1]
    xb1 = b[1,1]
    ya0 = a[0,0]
    yb0 = b[0,0]
    ya1 = a[1,0]
    yb1 = b[1,0]
    x_overlap = max(0, min(xa1, xb1) - max(xa0, xb0))
    y_overlap = max(0, min(ya1, yb1) - max(ya0, yb0))
    a_area = float((ya1 - ya0)*(xa1 - xa0))
    b_area = float((yb1 - yb0)*(xb1 - xb0))
    overlap_area = float(x_overlap * y_overlap)
    tp = float(x_overlap * y_overlap)
    fp = float(b_area - overlap_area)
    p = float(a_area)
    n = float(783*522 - p)
    if score == 'tpr' or score == 'recall':
        return tp/p
    elif score == 'fpr':
        return fp/n
    elif score == 'pascal':
        return overlap_area/(a_area + b_area - overlap_area)
    elif score == 'precision':
        return tp/(tp+fp)
