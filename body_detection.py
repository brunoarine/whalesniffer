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
IMG_DIR = '/data/images/'

def imhist(img, range=None):
    """Plot image histogram using 256 bins.
    """
    plt.hist(img.flatten(), 256, color='black', range=range)


# Load list of test images
filelist = glob.glob(IMG_DIR)
ground_truth = whalesniffer.load_annotations('data/annotations.json')

models = (
    (u'Clustering', whalesniffer.body.Clustering()),
    (u'Bayesian', whalesniffer.body.Bayesian()),
    (u'Manual', whalesniffer.body.Manual())
)

y_actual = [ground_truth[pathutils.strip_path(x)] for x in filelist]

index_names = [pathutils.strip_path(x) for x in filelist]
col_names = [x for (x, _) in models]

# Instantiate results data frames
df_iou = pd.DataFrame(index=index_names, columns=col_names)
df_recall = pd.DataFrame(index=index_names, columns=col_names)
df_precision = pd.DataFrame(index=index_names, columns=col_names)
df_fpr = pd.DataFrame(index=index_names, columns=col_names)
df_areas = pd.DataFrame(index=index_names, columns=col_names)

statistics = {}
for model_name, model in models:
    t_start = time.time()
    y_predicted = model.predict(filelist)
    t_end = time.time()
    t_total = t_end - t_start

    iou = [evaluate_score(y_actual[n], y_predicted[n], 'iou') for n in range(len(filelist))]
    recall = [evaluate_score(y_actual[n], y_predicted[n], 'tpr') for n in range(len(filelist))]
    precision = [evaluate_score(y_actual[n], y_predicted[n], 'precision') for n in range(len(filelist))]
    fpr = [evaluate_score(y_actual[n], y_predicted[n], 'fpr') for n in range(len(filelist))]

    df_iou[model_name] = iou
    df_recall[model_name] = recall
    df_precision[model_name] = precision
    df_fpr[model_name] = fpr
    df_areas[model_name] = y_predicted

    statistics[model_name] = {'total_time': t_total}

df_areas['Actual'] = y_actual

if SHOW_THUMBS:
    for model in df_iou:
        worst = df_iou[model].sort_values()[:2]

        for filename in worst.index:
            plt.figure(figsize=(10, 5))
            p0 = tuple(df_areas.ix[filename][model][0][::-1])
            p1 = tuple(df_areas.ix[filename][model][1][::-1])
            a0 = tuple(df_areas.ix[filename]['Actual'][0][::-1])
            a1 = tuple(df_areas.ix[filename]['Actual'][1][::-1])
            plt.figure()
            img = plt.imread(IMG_DIR + filename)
            cv2.rectangle(img, p0, p1, (1, 0, 0), 2)
            cv2.rectangle(img, a0, a1, (0, 0, 1), 2)
            plt.title(u'{}\n(overlap = {:.2f}, coverage = {:.2f})'
                      .format(model, df_iou.ix[filename][model], df_recall.ix[filename][model]))
            plt.imshow(img)

    for model in df_iou:
        worst = df_iou[model].sort_values(ascending=False)[:2]

        for filename in worst.index:
            plt.figure(figsize=(10, 5))
            p0 = tuple(df_areas.ix[filename][model][0][::-1])
            p1 = tuple(df_areas.ix[filename][model][1][::-1])
            a0 = tuple(df_areas.ix[filename]['Actual'][0][::-1])
            a1 = tuple(df_areas.ix[filename]['Actual'][1][::-1])
            plt.figure()
            img = plt.imread(IMG_DIR + filename)
            cv2.rectangle(img, p0, p1, (0, 1, 0), 2)
            cv2.rectangle(img, a0, a1, (0, 0, 1), 2)
            plt.title(u'{}\n(sobreposição = {:.2f}, abrangência = {:.2f})'
                      .format(model, df_iou.ix[filename][model], df_recall.ix[filename][model]))
            plt.imshow(img)

print(datetime.datetime.now())
print(u"{:<25} {:<25}".format('Model', 'Execution time'))
for model, stats in iter(statistics.items()):
    print(u"{:<25} {:<25}".format(model, stats['total_time']))

df_areas.to_csv('reports/df_areas.csv', encoding='utf-8')
df_recall.to_csv('reports/df_recall.csv', encoding='utf-8')
df_iou.to_csv('reports/df_iou.csv', encoding='utf-8')
df_precision.to_csv('reports/df_precision.csv', encoding='utf-8')
df_fpr.to_csv('reports/df_fpr.csv', encoding='utf-8')
