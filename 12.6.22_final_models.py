# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:36:41 2021

@author: makow
"""

from utils import *
from sklearn.metrics import roc_curve, auc, f1_score, recall_score, precision_score

#%%
data = pd.read_csv("tradeoff_model_moe_features_measured_reduced.csv", header = 0, index_col = 0)
target = pd.read_csv("7.12.21_data_targets.csv", header = 0, index_col = 0)
target['Combined Label'] = 1
target.loc[(target['SMP Score'] < 0.33) & (target['CS-SINS Score'] < 0.25), 'Combined Label'] = 0
target.loc[target['CS-SINS Score'] > 0.25, 'CS-SINS Label'] = 1
target.loc[target['SMP Score'] > 0.33, 'SMP Label'] = 1
target.loc[target['OVA Score'] > 0.11, 'OVA Label'] = 1

pan_data = pd.read_csv("tradeoff_model_moe_features_pan_variants.csv", header = 0, index_col = 0)
cin_data = pd.read_csv("tradeoff_model_moe_features_cinp_variants.csv", header = 0, index_col = 0)
gan_data = pd.read_csv("tradeoff_model_moe_features_gan_variants.csv", header = 0, index_col = 0)


#%%
### sins, app_charge, dipole_moment, patch_hyd
data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced = data_parse3(data, pan_data, cin_data, gan_data, 'app_charge', 'dipole_moment', 'patch_hyd_%')
clf = GNB()
data_predict, data_predict_proba = clf_model(clf, data_reduced, target.iloc[:,3])
print(accuracy_score(data_predict, target.iloc[:,3]))
print(balanced_accuracy_score(data_predict, target.iloc[:,3]))
print(f1_score(data_predict, target.iloc[:,3]))
print(recall_score(data_predict, target.iloc[:,3]))
print(precision_score(data_predict, target.iloc[:,3]))
pan_data_predict = clf.predict(pan_data_reduced)
cin_data_predict = clf.predict(cin_data_reduced)
gan_data_predict = clf.predict(gan_data_reduced)

#print(sc.stats.spearmanr(data_reduced[:,0], data_reduced[:,1]))
#plt.scatter(data_reduced[:,0], data_reduced[:,1])

pan_data_predict_proba = clf.predict_proba(pan_data_reduced)
cin_data_predict_proba = clf.predict_proba(cin_data_reduced)
gan_data_predict_proba = clf.predict_proba(gan_data_reduced)

print(sc.stats.spearmanr(data_predict_proba[:,0], target.iloc[:,0]))

gnb_prob_plot_sins(clf, data_reduced, target.iloc[:,3], cin_data_reduced, pan_data_reduced)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)

#%%
### smp, app_charge, dipole_moment, patch_hyd
data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced = data_parse3(data, pan_data, cin_data, gan_data, 'hyd_moment', 'app_charge', 'patch_hyd_%')
clf = GNB()
data_predict, data_predict_proba = clf_model(clf, data_reduced, target.iloc[:,4])
print(accuracy_score(data_predict, target.iloc[:,4]))
print(balanced_accuracy_score(data_predict, target.iloc[:,4]))
print(f1_score(data_predict, target.iloc[:,4]))
print(recall_score(data_predict, target.iloc[:,4]))
print(precision_score(data_predict, target.iloc[:,4]))
pan_data_predict = clf.predict(pan_data_reduced)
cin_data_predict = clf.predict(cin_data_reduced)
gan_data_predict = clf.predict(gan_data_reduced)

pan_data_predict_proba = clf.predict_proba(pan_data_reduced)
cin_data_predict_proba = clf.predict_proba(cin_data_reduced)
gan_data_predict_proba = clf.predict_proba(gan_data_reduced)

print(sc.stats.spearmanr(data_predict_proba[:,0], target.iloc[:,1]))

gnb_prob_plot_smp(clf, data_reduced, target.iloc[:,4], cin_data_reduced, pan_data_reduced)
plt.xticks(fontsize = 24)
plt.yticks(fontsize = 24)

