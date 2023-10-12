from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import seaborn as sns
import pylab
import scipy.stats as stats
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math


class Visualization():
    def __init__(self, savePath, n_cols=6, cmap=sns.diverging_palette(230, 20, as_cmap=True)):
        self.n_cols=n_cols
        self.cmap = cmap
        if savePath[-1] == '/' or savePath[-1] == '\"':
            self.savePath = savePath
        else:
            if "/" in savePath:
                self.savePath = savePath+"/"
            else:
                self.savePath = savePath+'\"'
    
    def show_distplots(self, df, columns, savePicture = False, pictureName= None): #columns == numeric_cols
        ncols = self.n_cols
        nrows = int(len(columns)/ncols+1)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 50), sharey=False)
        for i, target in enumerate(columns):
            ax = axs[int(i/ncols), i % ncols]
            _ = sns.histplot(df[target], kde=True, stat="density", kde_kws=dict(cut=3), alpha=.4, edgecolor=(1, 1, 1, .4), ax=ax)
            ax.set_title(target)
        plt.tight_layout()
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()

    def show_probplots(self, df, columns, savePicture = False, pictureName= None): #columns == numeric_cols
        ncols = self.n_cols
        nrows = int(len(columns)/ncols+1)

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 50), sharey=False)
        for i, target in enumerate(columns):
            ax = axs[int(i/ncols), i % ncols]
            stats.probplot(df[target], dist="norm", plot=ax)
            ax.set_title(target)
        plt.tight_layout()
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()

    def show_kde(self, df, columns, group, savePicture = False, pictureName= None):
        ncols = self.n_cols
        nrows = math.ceil(len(columns)/ncols)
        group_unique_values = list(df[group].unique())

        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(30, 50), sharey=False)
        for i, c in enumerate(columns):
            ax = axs[int(i/ncols), i % ncols]
            for value in group_unique_values:
                try:
                    df[df[group] == value][c].plot.kde(alpha=1.0/len(group_unique_values), label=value, ax=ax)
                except LA.LinAlgError:
                    pass
            ax.set_title(c)
            ax.legend(loc='upper right')
        plt.tight_layout()
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()

    def show_boxplot(self, df, columns, x, hue= None, savePicture= False, pictureName=None):
        nrows = int(len(columns)/self.ncols+1)
        fig, axs = plt.subplots(nrows=nrows, ncols=self.n_cols, figsize=(30, 60), sharey=False)
        for i, target in enumerate(columns):
            ax = axs[int(i/self.ncols), i % self.ncols]
            if hue != None:
                sns.stripplot(ax=ax, x=x, y=target, hue=hue, data=df, color='black', size=10, alpha=0.3)
                sns.boxplot(ax=ax, x=x, y=target, hue=hue, data=df, showmeans=True,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
            else:
                sns.stripplot(ax=ax, x=x, y=target, data=df, color='black', size=10, alpha=0.3)
                sns.boxplot(ax=ax, x=x, y=target, data=df, showmeans=True,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        plt.tight_layout()
        plt.show()
        pass

    def show_wordCloud(self):
        pass
    