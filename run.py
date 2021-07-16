# -*- coding: utf-8 -*-

import datetime
import glob
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

import whalesniffer.body
from whalesniffer.utils import evaluate_score
from whalesniffer.utils import pathutils

SAVE_FIGURES = True
DATA_DIR = 'data/images/raw/'
REPORT_IMG_DIR = 'reports/images/'
AREAS_FILENAME = "reports/areas.csv"
IOU_FILENAME = "reports/iou_score.csv"
PRECISION_FILENAME = "reports/precision_score.csv"
RECALL_FILENAME = "reports/recall_score.csv"
FPR_FILENAME = "reports/fpr_score.csv"


def save_result_img(image_names, label, model):
    """Plot selected cases.
    """
    for filenum, image_name in enumerate(image_names.index):
        p0 = tuple(df_areas.loc[image_name, model][0][::-1])
        p1 = tuple(df_areas.loc[image_name, model][1][::-1])
        a0 = tuple(df_areas.loc[image_name, "Actual"][0][::-1])
        a1 = tuple(df_areas.loc[image_name, "Actual"][1][::-1])

        fig, ax = plt.subplots()
        img = plt.imread(DATA_DIR + image_name)
        cv2.rectangle(img, p0, p1, (0, 255, 0), 2)
        cv2.rectangle(img, a0, a1, (0, 0, 255), 2)

        custom_lines = [Line2D([0], [0], color="green", lw=2),
                        Line2D([0], [0], color="blue", lw=2)]

        ax.legend(custom_lines, ['predicted', 'true'])

        plt.title(u'{}\n(IoU = {:.2f})'.format(model, df_iou.loc[image_name, model]))
        plt.imshow(img)
        plt.tight_layout()
        plt.savefig(f'{REPORT_IMG_DIR}{model}-{label}-{filenum}.png')


# Load list of test image_names
ground_truth = whalesniffer.load_annotations('data/annotations.json')

filelist = glob.glob(DATA_DIR + "*.*")
y_actual = [ground_truth[pathutils.strip_path(x)] for x in filelist]

models = (
    #(u'KMeans', whalesniffer.body.Clustering()),
    #(u'Manual', whalesniffer.body.Manual()),
    (u'HistSimilarity', whalesniffer.body.Similarity()),
    (u'Thresholding', whalesniffer.body.Thresholding()),
    (u'Superpixels', whalesniffer.body.Superpixel()),
)

index_names = [pathutils.strip_path(x) for x in filelist]
col_names = [x for (x, _) in models]

# Instantiate results data frames
df_iou = pd.DataFrame(index=index_names, columns=col_names)
df_recall = pd.DataFrame(index=index_names, columns=col_names)
df_precision = pd.DataFrame(index=index_names, columns=col_names)
df_fpr = pd.DataFrame(index=index_names, columns=col_names)
df_areas = pd.DataFrame(index=index_names, columns=col_names)

time_stats = {}
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

    time_stats[model_name] = {'total_time': t_total}

df_areas['Actual'] = y_actual

if SAVE_FIGURES:
    for model_name, _ in models:
        save_result_img(image_names=df_iou[model_name].sort_values()[:2], label="worst", model=model_name)
        save_result_img(image_names=df_iou[model_name].sort_values(ascending=False)[:2], label="best", model=model_name)

# Print execution time time_stats
print(datetime.datetime.now())
print(u"{:<25} {:<25} {:<25}".format('Model', 'Execution time', 'Avg. IoU'))
for model, stats in iter(time_stats.items()):
    print(u"{:<25} {:<25} {:<25}".format(model, stats['total_time'], df_iou[model].mean()))
print("")

print(f"Saving rectangles in {AREAS_FILENAME}")
df_areas.to_csv(AREAS_FILENAME, encoding='utf-8')

print(f"Saving intersection-over-union scores in {IOU_FILENAME}")
df_iou.to_csv(IOU_FILENAME, encoding='utf-8')

print(f"Saving recall scores in {RECALL_FILENAME}")
df_recall.to_csv(RECALL_FILENAME, encoding='utf-8')

print(f"Saving precision scores in {PRECISION_FILENAME}")
df_precision.to_csv(PRECISION_FILENAME, encoding='utf-8')

print(f"Saving false-positive rates in {FPR_FILENAME}")
df_fpr.to_csv(FPR_FILENAME, encoding='utf-8')
