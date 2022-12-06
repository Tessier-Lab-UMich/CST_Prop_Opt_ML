  # -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:36:41 2021

@author: makow
"""

from utils_variant import *
from scipy.stats import ttest_ind, anderson, anderson_ksamp

data = pd.read_csv("tradeoff_model_moe_features_measured_reduced.csv", header = 0, index_col = 0)
target = pd.read_csv("7.12.21_data_targets.csv", header = 0, index_col = 0)
target.loc[target['CS-SINS Score'] > 0.35, 'CS-SINS Label'] = 1
target.loc[target['SMP Score'] > 0.19, 'SMP Label'] = 1
target.loc[target['OVA Score'] > 0.11, 'OVA Label'] = 1

pan_data = pd.read_csv("tradeoff_model_moe_features_pan_variants.csv", header = 0, index_col = 0)
cin_data = pd.read_csv("tradeoff_model_moe_features_cinp_variants.csv", header = 0, index_col = 0)
gan_data = pd.read_csv("tradeoff_model_moe_features_gan_variants.csv", header = 0, index_col = 0)

#%%
from utils_variant import *
### sins, app_charge, dipole_moment, patch_hyd
data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced = data_parse3(data, pan_data, cin_data, gan_data, 'app_charge', 'dipole_moment', 'patch_hyd_%')
clf = GNB()
data_predict, data_predict_proba = clf_model(clf, data_reduced, target.iloc[:,3])
print(accuracy_score(data_predict, target.iloc[:,3]))
pan_data_predict = clf.predict(pan_data_reduced)
cin_data_predict = clf.predict(cin_data_reduced)
gan_data_predict = clf.predict(gan_data_reduced)

pan_data_predict_proba = clf.predict_proba(pan_data_reduced)
cin_data_predict_proba = clf.predict_proba(cin_data_reduced)
gan_data_predict_proba = clf.predict_proba(gan_data_reduced)

print(sc.stats.spearmanr(data_predict_proba[:,0], target.iloc[:,0]))
gnb_prob_plot_sins(clf, data_reduced, target, cin_data_reduced, cin_data_predict, gan_data_reduced, gan_data_predict, pan_data_reduced, pan_data_predict)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

#%%
from utils_variant import *
### smp, app_charge, dipole_moment, patch_hyd
data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced = data_parse3(data, pan_data, cin_data, gan_data, 'hyd_moment', 'app_charge', 'patch_hyd_%')
clf = GNB()
data_predict, data_predict_proba = clf_model(clf, data_reduced, target.iloc[:,4])
print(accuracy_score(data_predict, target.iloc[:,4]))
pan_data_predict = clf.predict(pan_data_reduced)
cin_data_predict = clf.predict(cin_data_reduced)
gan_data_predict = clf.predict(gan_data_reduced)

pan_data_predict_proba = clf.predict_proba(pan_data_reduced)
cin_data_predict_proba = clf.predict_proba(cin_data_reduced)
gan_data_predict_proba = clf.predict_proba(gan_data_reduced)

print(sc.stats.spearmanr(data_predict_proba[:,0], target.iloc[:,1]))

gnb_prob_plot_smp(clf, data_reduced, target, cin_data_reduced, cin_data_predict, gan_data_reduced, gan_data_predict, pan_data_reduced, pan_data_predict)
plt.xticks(fontsize = 18)
plt.yticks(fontsize = 18)

#%%
cin_results = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.7.20_tradeoffs_analysis\\10.21.21_cin_targets.csv", header = 0, index_col = 0)
pan_results = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.7.20_tradeoffs_analysis\\2.17.22_pan_targets.csv", header = 0, index_col = 0)
gan_results = pd.read_csv("C:\\Users\\makow\\Documents\\Research\\Data Analysis\\10.7.20_tradeoffs_analysis\\2.17.22_gan_targets.csv", header = 0, index_col = 0)

#%%
fig, ax = plt.subplots(figsize = (14,4))
ax.bar(np.arange(22)-0.2, list((cin_results.iloc[:,0]))+[0]+list((gan_results.iloc[:,0]))+[0]+list((pan_results.iloc[:,0])), color = 'blue', width = 0.4, edgecolor = 'k', linewidth = 0.25)
ax.scatter(np.arange(22)-0.2, list((cin_results.iloc[:,7]))+[0]+list((gan_results.iloc[:,7]))+[0]+list((pan_results.iloc[:,7])), s = 75, color = 'silver', edgecolor = 'blue', linewidth = 0.25, zorder = 2)
ax.scatter(np.arange(22)-0.2, list((cin_results.iloc[:,8]))+[0]+list((gan_results.iloc[:,8]))+[0]+list((pan_results.iloc[:,8])), s = 75, color = 'silver', edgecolor = 'blue', linewidth = 0.25, zorder = 2)
ax.scatter(np.arange(22)-0.2, list((cin_results.iloc[:,9]))+[0]+list((gan_results.iloc[:,9]))+[0]+list((pan_results.iloc[:,9])), s = 75, color = 'silver', edgecolor = 'blue', linewidth = 0.25, zorder = 2)

plt.xticks(np.arange(22))
plt.yticks([0.0,0.5, 1.0, 1.5],fontsize = 18)
ax2 = ax.twinx()
ax2.bar(np.arange(22)+0.2, list((cin_results.iloc[:,1]))+[0]+list((gan_results.iloc[:,1]))+[0]+list((pan_results.iloc[:,1])), color = 'red', width = 0.4, edgecolor = 'k', linewidth = 0.25)
ax2.scatter(np.arange(22)+0.2, list((cin_results.iloc[:,11]))+[0]+list((gan_results.iloc[:,11]))+[0]+list((pan_results.iloc[:,11])), s = 75, color = 'silver', edgecolor = 'red', linewidth = 0.25, zorder = 2)
ax2.scatter(np.arange(22)+0.2, list((cin_results.iloc[:,12]))+[0]+list((gan_results.iloc[:,12]))+[0]+list((pan_results.iloc[:,12])), s = 75, color = 'silver', edgecolor = 'red', linewidth = 0.25, zorder = 2)
ax2.scatter(np.arange(22)+0.2, list((cin_results.iloc[:,13]))+[0]+list((gan_results.iloc[:,13]))+[0]+list((pan_results.iloc[:,13])), s = 75, color = 'silver', edgecolor = 'red', linewidth = 0.25, zorder = 2)
plt.xticks(np.arange(22))
plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4], fontsize = 18)

#%%
fig, ax = plt.subplots(figsize = (14,4))
ax.bar(np.arange(22)-0.2, list((cin_results.iloc[:,2]))+[0]+list((gan_results.iloc[:,2]))+[0]+list((pan_results.iloc[:,2])), color = 'blue', width = 0.4, edgecolor = 'k', linewidth = 0.25)
ax.scatter(np.arange(22)-0.2, list((cin_results.iloc[:,15]))+[-2]+list((gan_results.iloc[:,15]))+[-2]+list((pan_results.iloc[:,15])), s = 75, color = 'silver', edgecolor = 'blue', linewidth = 0.25, zorder = 2)
ax.scatter(np.arange(22)-0.2, list((cin_results.iloc[:,16]))+[-2]+list((gan_results.iloc[:,16]))+[-2]+list((pan_results.iloc[:,16])), s = 75, color = 'silver', edgecolor = 'blue', linewidth = 0.25, zorder = 2)
ax.scatter(np.arange(22)-0.2, list((cin_results.iloc[:,17]))+[-2]+list((gan_results.iloc[:,17]))+[-2]+list((pan_results.iloc[:,17])), s = 75, color = 'silver', edgecolor = 'blue', linewidth = 0.25, zorder = 2)

plt.xticks(np.arange(22))
plt.ylim(0,1.0)
plt.yticks([0.0,0.2,0.4, 0.6, 0.8, 1.0],fontsize = 18)
ax2 = ax.twinx()
ax2.bar(np.arange(22)+0.2, list((cin_results.iloc[:,19]))+[0]+list((gan_results.iloc[:,19]))+[0]+list((pan_results.iloc[:,19])), color = 'red', width = 0.4, edgecolor = 'k', linewidth = 0.25)
ax2.scatter(np.arange(22)+0.2, list((cin_results.iloc[:,20]))+[0]+list((gan_results.iloc[:,20]))+[0]+list((pan_results.iloc[:,20])), s = 75, color = 'silver', edgecolor = 'red', linewidth = 0.25, zorder = 2)
ax2.scatter(np.arange(22)+0.2, list((cin_results.iloc[:,21]))+[0]+list((gan_results.iloc[:,21]))+[0]+list((pan_results.iloc[:,21])), s = 75, color = 'silver', edgecolor = 'red', linewidth = 0.25, zorder = 2)
plt.xticks(np.arange(22))
plt.ylim(46,76)
plt.yticks([50, 60, 70], fontsize = 18)


#%%
plt.figure(figsize = (4,4.8))
sns.swarmplot(target['Type'], target['Clearance'], s = 10, edgecolor = 'k', linewidth = 0.25)
plt.yticks(fontsize = 18)

#%%
plt.figure(figsize = (3.5, 3.8))
plt.bar(np.arange(3), [0, 70, 40], width = 0.6, color = 'blue', edgecolor = 'k', linewidth = 0.25)
plt.xticks(np.arange(3))
plt.yticks([0,20,40,60,80,100],fontsize = 20)

from scipy.stats import fisher_exact
oddsratio, pvalue = fisher_exact([[0,6], [7,3]])
oddsratio, pvalue = fisher_exact([[0,6], [4,6]])
oddsratio, pvalue = fisher_exact([[4,6], [7,3]])


#%%
plt.figure(figsize = (5, 3.8))
plt.bar(np.arange(4), [6, 10, 10, 0], width = 0.6, color = 'blue', edgecolor = 'k', linewidth = 0.25)
plt.xticks(np.arange(4))
plt.yticks([0,4,8,12],fontsize = 20)

#%%
plt.figure(figsize = (2.5,4.8))
target_group = target.groupby(['Type', 'VH v_gene simple', 'Type count']).size().reset_index()
plt.scatter(target_group.iloc[:,0], target_group.iloc[:,1], s = target_group.iloc[:,3]*100, c = target_group.iloc[:,3]/target_group['Type count'], cmap = 'bwr', edgecolor = 'k', linewidth = 0.5, vmin = 0, vmax = 0.75)
plt.xlim(-0.5,3.5)
plt.ylim(-0.5,6.5)

#%%
plt.figure(figsize = (2.5,4.8))
target_group = target.groupby(['Type', 'VL v_gene simple', 'Type count']).size().reset_index()
plt.scatter(target_group.iloc[:,0], target_group.iloc[:,1], s = target_group.iloc[:,3]*100, c = target_group.iloc[:,3]/target_group['Type count'], cmap = 'bwr', edgecolor = 'k', linewidth = 0.5, vmin = 0, vmax = 0.75)
plt.xlim(-0.5,3.5)
plt.ylim(-0.5,4.5)

#%%
plt.figure(figsize = (2.5,4.8))
target_group = target.groupby(['Type', 'Heavy chain', 'Type count']).size().reset_index()
plt.scatter(target_group.iloc[:,0], target_group.iloc[:,1], s = target_group.iloc[:,3]*100, c = target_group.iloc[:,3]/target_group['Type count'], cmap = 'bwr', edgecolor = 'k', linewidth = 0.5, vmin = 0, vmax = 0.75)
plt.xlim(-0.5,3.5)
plt.ylim(-0.5,4.5)



