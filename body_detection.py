# -*- coding: utf-8 -*-

import time
import datetime
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import whalesniffer.body
from whalesniffer.utils import pathutils
from whalesniffer.utils import evaluate_score


SHOW_THUMBS = True

def imhist(img, range=None):
    plt.hist(img.flatten(),256, color = 'black', range=range)

#directory = '/home/soldeace/Pictures/small/'
directory = 'c:\\temp\\small\\'
filelist = glob.glob(directory + 'w_23.png')
dict_actual = whalesniffer.load_annotations('data/annotations.json')
models = ( 
 (u'Clustering', whalesniffer.body.Clustering()),
 (u'Bayesian', whalesniffer.body.Bayesian()),
 (u'Manual', whalesniffer.body.Manual())

 )

y_actual = [dict_actual[pathutils.strip_path(x)] for x in filelist]

df_pascal = pd.DataFrame(index=[pathutils.strip_path(x) for x in filelist],
                         columns=[x for (x, _) in models])
df_recall = pd.DataFrame(index=[pathutils.strip_path(x) for x in filelist],
                         columns=[x for (x, _) in models])
df_precision = pd.DataFrame(index=[pathutils.strip_path(x) for x in filelist],
                         columns=[x for (x, _) in models])                         
df_fpr = pd.DataFrame(index=[pathutils.strip_path(x) for x in filelist],
                         columns=[x for (x, _) in models])                         
df_areas = pd.DataFrame(index=[pathutils.strip_path(x) for x in filelist],
                         columns=[x for (x, _) in models])
            

statistics = {}

for model_name, model in models:    
    
    t_start = time.clock()
    y_predicted = model.predict(filelist)
    t_end = time.clock()
    t_total = t_end - t_start
    
    
        
    pascal = [evaluate_score(y_actual[n], y_predicted[n], 'pascal') for n in range(len(filelist))]
    recall = [evaluate_score(y_actual[n], y_predicted[n], 'tpr') for n in range(len(filelist))]
    precision = [evaluate_score(y_actual[n], y_predicted[n], 'precision') for n in range(len(filelist))]
    fpr = [evaluate_score(y_actual[n], y_predicted[n], 'fpr') for n in range(len(filelist))]
    
    df_pascal[model_name] = pascal
    df_recall[model_name] = recall
    df_precision[model_name] = precision
    df_fpr[model_name] = fpr
    df_areas[model_name] = y_predicted
                                   
    statistics[model_name] = {'total_time': t_total}
    

df_areas['Actual'] = y_actual

#with open('reports/run {}.txt'.format(datetime.datetime.now()), 'wb') as handle:
#    pickle.dump(statistics,handle)


if SHOW_THUMBS == True:
    for model in df_pascal:
        worst = df_pascal[model].sort_values()[:2] 
        
        for filename in worst.index:
            plt.figure(figsize=(10,5))
            p0 = tuple(df_areas.ix[filename][model][0][::-1])
            p1 = tuple(df_areas.ix[filename][model][1][::-1])
            a0 = tuple(df_areas.ix[filename]['Actual'][0][::-1])
            a1 = tuple(df_areas.ix[filename]['Actual'][1][::-1])            
            plt.figure()    
            img = plt.imread(directory + filename)
            cv2.rectangle(img, p0, p1, (1,0,0), 2)
            cv2.rectangle(img, a0, a1, (0,0,1), 2)
            plt.title(u'{}\n(sobreposição = {:.2f}, abrangência = {:.2f})'.format(model,df_pascal.ix[filename][model],df_recall.ix[filename][model]))
            plt.imshow(img)
            

if SHOW_THUMBS == True:
    for model in df_pascal:
        worst = df_pascal[model].sort_values(ascending=False)[:2] 
        
        for filename in worst.index:
            plt.figure(figsize=(10,5))
            p0 = tuple(df_areas.ix[filename][model][0][::-1])
            p1 = tuple(df_areas.ix[filename][model][1][::-1])
            a0 = tuple(df_areas.ix[filename]['Actual'][0][::-1])
            a1 = tuple(df_areas.ix[filename]['Actual'][1][::-1])            
            plt.figure()    
            img = plt.imread(directory + filename)
            cv2.rectangle(img, p0, p1, (0,1,0), 2)
            cv2.rectangle(img, a0, a1, (0,0,1), 2)
            plt.title(u'{}\n(sobreposição = {:.2f}, abrangência = {:.2f})'.format(model,df_pascal.ix[filename][model],df_recall.ix[filename][model]))
            plt.imshow(img)


print datetime.datetime.now()
print u"{:<25} {:<25}".format('Modelo','Demora')
for model, stats in statistics.iteritems():    
    print u"{:<25} {:<25}".format(model, stats['total_time'])
    #np.sum(stats['pascal'] >= 0.5)/float(len(stats['pascal'])) )
    
df_areas.to_csv('reports\\df_areas.csv', encoding='utf-8')
df_recall.to_csv('reports\\df_recall.csv', encoding='utf-8')
df_pascal.to_csv('reports\\df_pascal.csv', encoding='utf-8')
df_precision.to_csv('reports\\df_precision.csv', encoding='utf-8')
df_fpr.to_csv('reports\\df_fpr.csv', encoding='utf-8')