# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 09:32:05 2015

@author: soldeace
"""
import json
import numpy as np

def load_annotations(filename):
    with open(filename) as data_file:
        table = {}
        data = json.load(data_file)        
        for item in data:
            dirsep = '\\' if '\\' in item['filename'] else '/'            
            name = item['filename'].split(dirsep)[-1]
            value = item['annotations'][0]
            p0 = [value['y'], value['x']]
            p1 = [value['y'] + value['height'], value['x'] + value['width']]
            table[name] = np.array([p0,p1]).astype('int')
        return table