# -*- coding: utf-8 -*-

import time
import datetime
import glob
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import whalesniffer.body
from whalesniffer.utils import pathutils
from whalesniffer.utils import evaluate_score

SAVE_FIGURES = True
IMG_DIR = 'data/images/raw/'
REPORT_DIR = 'reports/images/'

def imhist(img, range=None):
    """Plot image histogram using 256 bins.
    """
    plt.hist(img.flatten(), 256, color='black', range=range)


def save_result_img(image_names, label):
    """Plot selected cases.
    """
    for filenum, image_name in enumerate(image_names.index):
        p0 = tuple(df_areas.loc[image_name, model][0][::-1])
        p1 = tuple(df_areas.loc[image_name, model][1][::-1])
        a0 = tuple(df_areas.loc[image_name, "Actual"][0][::-1])
        a1 = tuple(df_areas.loc[image_name, "Actual"][1][::-1])

        fig, ax = plt.subplots()
        img = plt.imread(IMG_DIR + image_name)
        cv2.rectangle(img, p0, p1, (0, 255, 0), 2)
        cv2.rectangle(img, a0, a1, (0, 0, 255), 2)

        custom_lines = [Line2D([0], [0], color="green", lw=2),
                        Line2D([0], [0], color="blue", lw=2)]

        ax.legend(custom_lines, ['predicted', 'true'])

        plt.title(u'{}\n(overlap = {:.2f}, coverage = {:.2f})'
                  .format(model, df_iou.loc[image_name, model], df_recall.loc[image_name, model]))
        plt.imshow(img)
        plt.tight_layout()
        plt.savefig(f'{REPORT_DIR}{label}_{filenum}.png')

# Load list of test image_names
ground_truth = whalesniffer.load_annotations('data/annotations.json')

filelist = glob.glob(IMG_DIR + "*.*")
y_actual = [ground_truth[pathutils.strip_path(x)] for x in filelist]

models = (
    (u'Clustering', whalesniffer.body.Clustering()),
    #(u'Bayesian', whalesniffer.body.Bayesian()),
    #(u'Manual', whalesniffer.body.Manual())
)

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

if SAVE_FIGURES:
    for model in df_iou:
        save_result_img(image_names=df_iou[model].sort_values()[:2], label="worst")
        save_result_img(image_names=df_iou[model].sort_values(ascending=False)[:2], label="best")

# Print execution time statistics
print(datetime.datetime.now())
print(u"{:<25} {:<25}".format('Model', 'Execution time'))
for model, stats in iter(statistics.items()):
    print(u"{:<25} {:<25}".format(model, stats['total_time']))

df_areas.to_csv('reports/df_areas.csv', encoding='utf-8')
df_recall.to_csv('reports/df_recall.csv', encoding='utf-8')
df_iou.to_csv('reports/df_iou.csv', encoding='utf-8')
df_precision.to_csv('reports/df_precision.csv', encoding='utf-8')
df_fpr.to_csv('reports/df_fpr.csv', encoding='utf-8')
