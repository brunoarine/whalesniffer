# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 21:32:07 2015

@author: barine
"""

import numpy as np
from sklearn import svm
from sklearn import neighbors
from sklearn import naive_bayes
from sklearn import ensemble
from sklearn import tree
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve, auc, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold

from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy import interp
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#############################################################################
# VARIAVEIS PRIMARIAS DE ENTRADA
#############################################################################
SUMMARY_FILE = 'params_semlbp_limpo.csv'
#SUMMARY_FILE = 'params_compareA.csv' #nem tanto
 #'params_semlbp_limpo.csv' #funcionou bem
RANDOM_STATE = 0


#############################################################################
# Extração das contagens do canais e inserção nas matrizes
#############################################################################
summary_matrix = np.genfromtxt(SUMMARY_FILE, skip_header=1, delimiter=',')

# 1   l1
# 2	l2
# 3	l3
# 4	l4
# 5	l5
# 6	h1
# 7	h2
# 8	h3
# 9	h4
#10	h5
#11	h6
#12	h7
#13	contrast
#14	correlation
#15   energy
#16	homogeneity


interest = [4,5,13,14]
interestB = [2,4,13,14]
interestC = [4,5,9,15,16]
interest_mean = [2,4,5,9,13,14,15,16]
interest_apresentacao = [5,13,14]
y = summary_matrix[:,0]

x = summary_matrix[:,interest_apresentacao]
#x = summary_matrix[:,1:]


X = StandardScaler().fit_transform(x)


###############################################################################
# Classification and ROC analysis

# Run classifier with cross-validation and plot ROC curves

cv = StratifiedKFold(y, n_folds=6)
#classifier_stupid = ensemble.AdaBoostClassifier()
classifier_stupid = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(max_features=2, min_samples_split=5, criterion='entropy', max_depth=8, min_samples_leaf=3))
# tree.DecisionTreeClassifier(max_features=2, min_samples_split=5, criterion='entropy', max_depth=8, min_samples_leaf=3)
# tree.DecisionTreeClassifier(max_features=2, min_samples_split=7, criterion='gini', max_depth=6, min_samples_leaf=3)




mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

precision = []
importancias = []
for i, (train, test) in enumerate(cv):
    classifier_stupid.fit(X[train],y[train])
    importancias.append(classifier_stupid.feature_importances_)
    classifier = CalibratedClassifierCV(classifier_stupid, method='sigmoid')    
    probas_ = classifier.fit(X[train], y[train]).predict_proba(X[test])
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    roc_auc = auc(fpr, tpr)
    y_pred = classifier.predict(x[test])    
    precision.append(metrics.f1_score(y[test], y_pred))
    #plt.plot(fpr, tpr, lw=1, label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

plt.figure(figsize=(10,6))
plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')

mean_tpr /= len(cv)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
plt.plot(mean_fpr, mean_tpr, '-',
         label='Boosted Decision Trees (area = %0.2f)' % mean_auc, lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig('curva_roc.png')

plt.show()

print np.array(importancias).mean(axis=0)
print np.array(importancias).std(axis=0)

fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1], X[:,2], c=y)
plt.xlabel('LBP (banda 5)')
plt.ylabel('Contraste')
plt.savefig('scatter3d.png')