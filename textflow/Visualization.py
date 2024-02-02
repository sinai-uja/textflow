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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import NMF
from sklearn import preprocessing as pre



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

    def show_wordCloud(self,df,textColumns, stopwords= None, titleGraphic= "+frecuentes", widthGraphic = 10,heightGraphic=6, hspace= 0.5,titleGraphicSize= 7, groupby=None, savePicture= False, pictureName=None):
        """
        Function that show wordcloud, opcionlly save this plot.

        Args:
            df: the pandas DataFrame that contains the data to visualize. This dataframe must to be filteren before pass for it
            textColumns: a list with the name of the columns of the dataframe that contain the sample data (text or a dictionary frequence) from which the wordCloud is created.
            stopwords: a list with the stopwords
            titleGraphic: a string with the title of the wordcloud. By defect it will be "+ frecuentes"
            widthGraphic: an integer that represent the width of the graphic
            heightGraphic: an integer that represent the height of the graphic
            hspace: the height of the padding between subplots, as a fraction of the average Axes height.
            titleGraphicSize : a integer that indicates the size of the title graphic
            groupby: a specific column of a DataFrame. The purpose of this variable is to indicate which column in the DataFrame will be used to group the data before performing the histogram.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A WordCloud shows in the notebook or an imagen saved in the corresponding path.
        """
        for numTc, tc in enumerate(textColumns):
            if type(df[tc].iloc[0]) == dict:
                plt.subplots_adjust(hspace=hspace)
                plt.figure(figsize=(widthGraphic,heightGraphic))
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
                            ncols = len(df[groupby[0]].unique())
                            nrows = math.ceil(len(df)/ncols)
                            plt.subplot(nrows,ncols, int(i)+1)
                            plt.imshow(WordCloud(background_color='white', width=1000, height=1000)
                                        .fit_words(r[tc]))
                            plt.axis("off")
                            plt.title(f"{tc} {titleGraphic} \n ({stringGroupBy})",fontsize=titleGraphicSize)
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
                            plt.title(f"{tc} {titleGraphic}",fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()
            else: #Es texto
                if groupby != None:
                    for i, row in df.groupby(groupby)[tc].agg(lambda x: ' '.join(x)).reset_index().iterrows():
                        stringGroupBy = row[groupby]
                        if type(groupby) == list:
                            groupColumnValues= [r[gb] for gb in groupby]
                            stringGroupBy = ', '.join(groupColumnValues)
                        ncols = len(textColumns)
                        nrows = math.ceil(len(df.groupby(groupby)[tc].agg(lambda x: ' '.join(x)).reset_index())/ncols)
                        plt.subplot(nrows,ncols, int(i)+1)
                        plt.imshow(WordCloud(background_color='white', width=1000, stopwords=stopwords, height=1000).generate(row[tc]))
                        plt.axis("off")
                        plt.title(f"{tc} {titleGraphic} ({stringGroupBy})", fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()
                else:
                    textWc=' '.join(list(df[tc]))
                    ncols = len(textColumns)
                    nrows = 1
                    plt.subplot(nrows,ncols, int(numTc)+1)
                    plt.imshow(WordCloud(background_color='white', width=500, stopwords = stopwords, height=600)
                                .generate(textWc))
                    plt.axis("off")
                    plt.title(f"{tc} {titleGraphic}",fontsize=titleGraphicSize)
                    if savePicture:
                        plt.savefig(self.savePath+pictureName)
                    plt.show()

    def show_pca(self, X, y, labelColumn= None, biplot = True, palette=None, savePicture= False, pictureName=None):
        """
        Function that show Principal Component Analysis graphic.

        Args:
            X: a pandas DataFrame with numeric values.
            y: a list with the labels associated to each row of X.
            labelColumn: a string with the name of the label column of X DataFrame. 
            biplot: bolean that indicate if a biplot graphic is shown. A biplot includes both the scatter plot and arrows indicating the direction and magnitude of the original features in the reduced space.
            palette: a string, list, dict, or matplotlib.colors.Colormap Method for choosing the colors to use when mapping the hue semantic.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A PCA graphic shows in the notebook or an imagen saved in the corresponding path.
            A a DataFrame that contains the principal component analysis (PCA) results.
        """
        scaler = StandardScaler()
        x = scaler.fit_transform(X)
        pca = PCA(n_components=2)
        components = pca.fit_transform(x)
        pca.components_

        #Varianza explicada
        cumVar = pd.DataFrame(np.cumsum(pca.explained_variance_ratio_)*100,
                            columns=["cumVarPerc"])
        expVar = pd.DataFrame(pca.explained_variance_ratio_*100, columns=["VarPerc"])
        pd.concat([expVar, cumVar], axis=1)\
            .rename(index={0: "PC1", 1: "PC2"})

        #Visualización
        componentsDf = pd.DataFrame(data = components, columns = ['PC1', 'PC2'])
        pcaDf = pd.concat([componentsDf, y], axis=1)

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=pcaDf, x="PC1", y="PC2", hue=labelColumn,palette=palette)
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()
        if biplot == True:
            score = components
            coeff = np.transpose(pca.components_)
            labels = list(X.columns)
            plt.figure(figsize=(12, 6))
            xs = score[:,0]
            ys = score[:,1]
            n = coeff.shape[0]
            scalex = 1.0/(xs.max() - xs.min())
            scaley = 1.0/(ys.max() - ys.min())
            plt.scatter(xs * scalex,ys * scaley,s=5)
            for i in range(n):
                plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
                if labels is None:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
                else:
                    plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

            plt.xlabel("PC{}".format(1))
            plt.ylabel("PC{}".format(2))
            plt.grid()
            plt.show()
        return pcaDf

    def show_tsne(self, X, tsne=TSNE(random_state = 0), label = None, savePicture= False, pictureName=None):
        """
        Function that show t-Distributed Stochastic Neighbor Embedding graphic.

        Args:
            X: a pandas DataFrame with the input data needs to be transformed and visualized t-SNE graphic.
            tsne:  The t-SNE algorithm.
            label: a list or pandas Series with the associated label of each row of X. 
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A TNSE graphic shows in the notebook or an imagen saved in the corresponding path.
        """
        data_t = tsne.fit_transform(X)
        sns.scatterplot(x=data_t[:, 0], y=data_t[:, 1], hue = label)
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()

    def show_nnmf(self, k, X, labels = None, iters=100, verbose=True, savePicture= False, pictureName=None):
        """
        Function that show Non-Negative Matrix Factorization graphic.

        Args:
            k: The number of components to factorize the input matrix X into.
            X: A pandas DataFrame with the input data to be factorized.
            labels: a list or pandas.Series with the labels for coloring the points in the scatter plot.
            iters: The number of iterations for the NNMF algorithm.
            verbose: A bolean that if is equal to True, print the relative error at every 10 iterations during the NNMF process.
            savePicture: a boolean value. True indicate that you want to save the pictura and False that the picture will not save. 
            pictureName: the name of the picture in the save path. By defect is None, because if the picture will not save, this variable will not be used.
        Returns:
            A NNMF graphic shows in the notebook or an imagen saved in the corresponding path.
            The factorized matrices W and H
            An array with the relative error at each iteration.
        """
        
        X = pre.MinMaxScaler().fit_transform(X)
        X = X/X.std(axis=0)
        # if X has any negative entries, stop:
        if (X < 0).any():
            raise ValueError('X must be non-negative.')

        n, dims = X.shape
        # initialize W, the measurement by cluster matrix
        W = np.random.uniform(0, 1, (n, k))
        # initialize H, the cluster by genes matrix
        H = np.random.uniform(0, 1, (k, dims))
        # initialize a vector to store errors
        error = np.empty(iters)

        # iteratively update W and H
        for i in np.arange(iters):
            newH = np.empty(H.shape)
            # calculate W.T * X (matrix multiply)
            N = np.dot(W.transpose(), X)
            # calculate W.T * W * H (matrix multiply)
            D = np.dot(np.dot(W.transpose(), W), H)
            # calculate H*N/D by element-wise multiplication
            newH = np.multiply(H, np.multiply(N, 1/D))
            H=newH.copy()
            # initialize a new W
            newW = np.empty(W.shape)
            N = X.dot(H.T)
            D = W.dot(H).dot(H.T)
            # element-wise multiply
            newW = np.multiply(W, np.multiply(N, 1/D))
            W = newW

            e = np.sqrt(np.sum((X - W.dot(H))**2))
            e_rel = e/np.sqrt(np.sum(X**2))
            error[i] = e_rel
            if not verbose:
                continue
            if i % 10 == 0:
                print('error is {0:.2f}'.format(e_rel))
        plt.figure()
        sns.scatterplot(x=W[:, 0], y=W[:, 1], hue = labels)
        plt.show()

        plt.figure()
        plt.plot(error)
        plt.xlabel('iterations')
        plt.ylabel('relative error')
        plt.show()
        sns.clustermap(W)
        if savePicture:
            plt.savefig(self.savePath+pictureName)
        plt.show()
        return W, H, error
