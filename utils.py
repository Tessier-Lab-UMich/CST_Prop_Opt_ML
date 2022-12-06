# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 19:41:56 2021

@author: makow
"""

import numpy as np
import pandas as pd
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessClassifier as GPC
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler
import itertools
from sklearn.model_selection import cross_validate as cv
from numpy import inf
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
cmap = plt.cm.get_cmap('bwr')
colormap= np.array([cmap(0.8),cmap(0.25)])
cmap1 = LinearSegmentedColormap.from_list("mycmap", colormap)

scaler = MinMaxScaler()

def data_parse2(data, pan_data, cin_data, gan_data, feature1, feature2):
    data_reduced = scaler.fit_transform(pd.concat([data[feature1], data[feature2]], axis = 1))
    pan_data_reduced = scaler.transform(pd.concat([pan_data[feature1], pan_data[feature2]], axis = 1))
    cin_data_reduced = scaler.transform(pd.concat([cin_data[feature1], cin_data[feature2]], axis = 1))
    gan_data_reduced = scaler.transform(pd.concat([gan_data[feature1], gan_data[feature2]], axis = 1))
    return data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced

def data_parse3(data, pan_data, cin_data, gan_data, feature1, feature2, feature3):
    data_reduced = scaler.fit_transform(pd.concat([data[feature1], data[feature2]/data[feature3]], axis = 1))
    pan_data_reduced = scaler.transform(pd.concat([pan_data[feature1], pan_data[feature2]/pan_data[feature3]], axis = 1))
    cin_data_reduced = scaler.transform(pd.concat([cin_data[feature1], cin_data[feature2]/cin_data[feature3]], axis = 1))
    gan_data_reduced = scaler.transform(pd.concat([gan_data[feature1], gan_data[feature2]/gan_data[feature3]], axis = 1))
    return data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced

def data_parse4(data, pan_data, cin_data, gan_data, feature1, feature2, feature3, feature4):
    data_reduced = scaler.fit_transform(pd.concat([data[feature1]/data[feature2], data[feature3]/data[feature4]], axis = 1))
    pan_data_reduced = scaler.transform(pd.concat([pan_data[feature1]/pan_data[feature2], pan_data[feature3]/pan_data[feature4]], axis = 1))
    cin_data_reduced = scaler.transform(pd.concat([cin_data[feature1]/cin_data[feature2], cin_data[feature3]/cin_data[feature4]], axis = 1))
    gan_data_reduced = scaler.transform(pd.concat([gan_data[feature1]/gan_data[feature2], gan_data[feature3]/gan_data[feature4]], axis = 1))
    return data_reduced, pan_data_reduced, cin_data_reduced, gan_data_reduced


def clf_model(clf, data, target):
    clf.fit(data, target)
    data_predict = clf.predict(data)
    data_predict_proba = clf.predict_proba(data)
    return data_predict, data_predict_proba

def scatter_hist(x, y, ax, ax_histx, ax_histy, target):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x, y, s=50, c=target, cmap='bwr', edgecolors=(0, 0, 0), linewidth = 0.25)

    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = 1

    bins = 15
    sns.distplot(x[target==0], bins=bins, color = 'blue', ax = ax_histx, kde_kws={"alpha": 0.2}, hist_kws={ "alpha": 0.2})
    sns.distplot(x[target==1], bins=bins, color = 'red', ax = ax_histx, kde_kws={"alpha": 0.2}, hist_kws={ "alpha": 0.2})
    ax_histx.set_xlim(-0.025,1.025)
    sns.distplot(y[target==0], bins=bins, color = 'blue', ax = ax_histy, vertical = True, kde_kws={"alpha": 0.2}, hist_kws={ "alpha": 0.2})
    sns.distplot(y[target==1], bins=bins, color = 'red', ax = ax_histy, vertical = True, kde_kws={"alpha": 0.2}, hist_kws={ "alpha": 0.2})
    ax_histy.set_ylim(-0.025,1.025)
    
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.01
rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

def gnb_prob_plot_sins(clf, data_scaled, target, pan_data_scaled, cin_data_scaled):
    xx, yy = np.meshgrid(np.linspace(-0.05, max(data_scaled[:,0])+0.05, 500), np.linspace(-0.05,  max(data_scaled[:,1])+0.05, 500))
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)

    image = ax.imshow(Z, interpolation='nearest',
                           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                           aspect='auto', origin='lower', cmap='bwr', alpha = 0.5, vmin = 0, vmax = 1)
    contours = ax.contour(xx, yy, Z, levels=[0.5], linewidths=1,
                               colors=['k'])
    scatter_hist(data_scaled[:,0], data_scaled[:,1], ax, ax_histx, ax_histy, target)
    ax.set_xticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 24)
    ax.set_xlim(-0.025,1.025)
    ax.set_yticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 24)
    ax.set_ylim(-0.025,1.025)
    plt.show()

def gnb_prob_plot_smp(clf, data_scaled, target, pan_data_scaled, cin_data_scaled):
    xx, yy = np.meshgrid(np.linspace(-0.05, max(data_scaled[:,0])+0.05, 500), np.linspace(-0.05,  max(data_scaled[:,1])+0.05, 500))
    Z = clf.predict_proba(np.vstack((xx.ravel(), yy.ravel())).T)[:, 1]
    Z = Z.reshape(xx.shape)
    fig = plt.figure(figsize = (8,6))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    image = ax.imshow(Z, interpolation='nearest',
                           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                           aspect='auto', origin='lower', cmap='bwr', alpha = 0.5, vmin = 0, vmax = 1)
    contours = ax.contour(xx, yy, Z, levels=[0.5], linewidths=1,
                               colors=['k'])
    scatter_hist(data_scaled[:,0], data_scaled[:,1], ax, ax_histx, ax_histy, target)
    ax.set_xticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 24)
    ax.set_xlim(-0.025,1.025)
    ax.set_yticklabels([0, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize = 24)
    ax.set_ylim(-0.025,1.025)
    plt.show()


    
