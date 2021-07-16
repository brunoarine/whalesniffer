# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 12:15:45 2015

@author: soldeace
"""

def strip_path(filename):
    dirsep = '\\' if '\\' in filename else '/'
    return filename.split(dirsep)[-1]