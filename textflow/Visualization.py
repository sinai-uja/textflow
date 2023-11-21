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
from wordcloud import WordCloud, STOPWORDS



class Visualization():
    """
    A class that provides methods to visualize different results of the analisys of a sequence.

    Attributes:
        savePath: an string with the path where different graphics where be saved.
        n_cols: the number of columns of the subplot grid.
        cmap: a colormap that is apply to the images.
    """
    def __init__(self, savePath='.', n_cols=6, cmap=sns.diverging_palette(230, 20, as_cmap=True)):
        """
    Create the class Visualization.

    Attributes:
        savePath: an string with the path where different graphics where be saved.
        n_cols: the number of columns of the subplot grid.
        cmap: a colormap that is apply to the images.
    """
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
        """
        Function that shows a histogram (using density normalization) and a superimposed kernel density estimated.

        Args:
            df: the pandas DataFrame that contains the data to visualize
            columns: a list with the name of the columns of the dataframe that contain the sample data from which the plot is created
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A density graphic shows in the notebook or an imagen saved in the corresponding path.
        """
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
        """
        Function that calculate quantiles for a probability plot, and show the plot or save this plot.

        Args:
            df: the pandas DataFrame that contains the data to visualize
            columns: a list with the name of the columns of the dataframe that contain the sample data from which the plot is created.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A probability plot shows in the notebook or an imagen saved in the corresponding path.
        """
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
        """
        Function that show kernel density estimate (KDE) plot, opcionlly save this plot.

        Args:
            df: the pandas DataFrame that contains the data to visualize
            columns: a list with the name of the columns of the dataframe that contain the sample data from which the plot is created.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A KDE plot shows in the notebook or an imagen saved in the corresponding path.
        """
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
        """
        Function that show box plot, opcionlly save this plot.

        Args:
            df: the pandas DataFrame that contains the data to visualize
            columns: a list with the name of the columns of the dataframe that contain the sample data from which the plot is created.
            x: name of variable in the DataFrame. This variable represent the input for plotting long-form data.
            hue: name of variable in the DataFrame. This variable represent the input for plotting long-form data.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A box plot shows in the notebook or an imagen saved in the corresponding path.
        """
        nrows = int(len(columns)/self.n_cols+1)
        fig, axs = plt.subplots(nrows=nrows, ncols=self.n_cols, figsize=(30, 60), sharey=False)
        for i, target in enumerate(columns):
            ax = axs[int(i/self.n_cols), i % self.n_cols]
            if hue != None:
                sns.stripplot(ax=ax, x=x, y=target, hue=hue, data=df, palette='dark:black', size=10, alpha=0.3)
                sns.boxplot(ax=ax, x=x, y=target, hue=hue, data=df, showmeans=True,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
            else:
                sns.stripplot(ax=ax, x=x, y=target, hue=x, legend=False, data=df, palette='dark:black', size=10, alpha=0.3)
                sns.boxplot(ax=ax, x=x, y=target, hue=x, legend=False, data=df, showmeans=True,
                            meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick'))
        plt.tight_layout()
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()
        

    def show_hist(self, df, columns, group, savePicture= False, pictureName=None):
        """
        Function that show histogram plot, opcionlly save this plot.

        Args:
            df: the pandas DataFrame that contains the data to visualize
            columns: a list with the name of the columns of the dataframe that contain the sample data from which the plot is created.
            group: a specific column of a DataFrame. The purpose of this variable is to indicate which column in the DataFrame will be used to group the data before performing the histogram.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A box plot shows in the notebook or an imagen saved in the corresponding path.
        """
        groups = list(df[group].unique())
        for nc in columns:
            plt.figure(figsize=(8,7))
            for c in groups:
                plt.hist(df[df[group] == c][nc], label=c, alpha=0.25)
            plt.legend(loc='upper right')
            plt.title(nc)
            plt.tight_layout()
            if savePicture:
                plt.savefig(self.savePath+pictureName)
            plt.show()

    def show_wordCloud(self,df,textColumns, stopwords= None, titleGraphic= None, hspace= 0.5,titleGraphicSize= 7, groupby=None, savePicture= False, pictureName=None):
        """
        Function that show wordcloud, opcionlly save this plot.

        Args:
            df,textColumns, stopwords= None, titleGraphic= None, groupby
            df: the pandas DataFrame that contains the data to visualize. This dataframe must to be filteren before pass for it
            textColumns: a list with the name of the columns of the dataframe that contain the sample data (text or a dictionary frequence) from which the wordCloud is created.
            stopwords: a list with the stopwords
            titleGraphic: a string with the title of the wordcloud
            titleGraphicSize : a integer that indicates the size of the title graphic
            groupby: a specific column of a DataFrame. The purpose of this variable is to indicate which column in the DataFrame will be used to group the data before performing the histogram.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A WordCloud shows in the notebook or an imagen saved in the corresponding path.
        """
        for tc in textColumns:
            if type(df[tc].iloc[0]) == dict:
                plt.subplots_adjust(hspace=hspace)
                if groupby != None:
                    for i, r in df.reset_index().iterrows():
                        if len(r[tc])> 0:
                            if stopwords != None:
                                for key in r[tc]:
                                    if key in stopwords:
                                        del r[tc][key]
                            stringGroupBy= groupby
                            if type(groupby) == list:
                                groupColumnValues= [r[gb] for gb in groupby]
                                stringGroupBy = ', '.join(groupColumnValues)
                            ncols = len(textColumns)
                            nrows = math.ceil(len(df)/ncols)
                            plt.subplot(nrows,ncols, int(i)+1)
                            plt.imshow(WordCloud(background_color='white', width=500, height=600)
                                        .fit_words(r[tc]))
                            plt.axis("off")
                            plt.title(f"{tc} +frecuentes ({stringGroupBy})",fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()
                else:
                    for i, r in df.reset_index().iterrows():
                        if len(r[tc]) > 0:
                            if stopwords != None:
                                for key in r[tc]:
                                    if key in stopwords:
                                        del r[tc][key]
                            ncols = len(textColumns)
                            nrows = math.ceil(len(df)/ncols)
                            plt.subplot(nrows,ncols, int(i)+1)
                            plt.imshow(WordCloud(background_color='white', width=500, height=600)
                                        .fit_words(r[tc]))
                            plt.axis("off")
                            plt.title(f"{tc} +frecuentes",fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()
            else: #Es texto
                if groupby != None:
                    for i, r in df.reset_index().iterrows():
                        stringGroupBy= groupby
                        if type(groupby) == list:
                            groupColumnValues= [r[gb] for gb in groupby]
                            stringGroupBy = ', '.join(groupColumnValues)
                        ncols = len(textColumns)
                        nrows = math.ceil(len(df)/ncols)
                        plt.subplot(nrows,ncols, int(i)+1)
                        plt.imshow(WordCloud(background_color='white', width=500, stopwords = stopwords, height=600)
                                    .generate(r[tc]))
                        plt.axis("off")
                        plt.title(f"{tc} +frecuentes ({stringGroupBy})",fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()
                else:
                    for i, r in df.reset_index().iterrows():
                        ncols = len(textColumns)
                        nrows = math.ceil(len(df)/ncols)
                        plt.subplot(nrows,ncols, int(i)+1)
                        plt.imshow(WordCloud(background_color='white', width=500, stopwords = stopwords, height=600)
                                    .generate(r[tc]))
                        plt.axis("off")
                        plt.title(f"{tc} +frecuentes",fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()
