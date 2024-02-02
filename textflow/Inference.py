from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SequentialFeatureSelector
from time import time
from sklearn.linear_model import RidgeCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
from IPython.display import display
from xgboost import XGBClassifier

class Inference():
    """
    A class that provides methods to provide inference methods with the analysis results.

    Attributes:
        classifiers: a list of tuples, where is tuple is formed by the name of the classifier to use and the classifier.

    """
    def __init__(self, classifiers= None):
        """
        Create the Inference class.

        Attributes:
            classifiers: a list of tuples, where is tuple is formed by the name of the classifier to use and the classifier. 
                         An example is [('Logistic regression', LogisticRegression(max_iter=1000))]
        """
        if classifiers == None:
            self.clfs = [('Logistic regression', LogisticRegression(max_iter=1000)),
                ('SVM', SVC()),
                ('Decision tree', DecisionTreeClassifier()),
                ('RandomForest', RandomForestClassifier(n_estimators=20, random_state=45)),
                ('SGD', SGDClassifier(max_iter=1000, tol=1e-4, random_state=45)),
                ('KNN', KNeighborsClassifier()),
                ('MLP', MLPClassifier(max_iter=1000)),
                ('XGBOOST', XGBClassifier(objective='binary:logistic',
                                            alpha=10,
                                            learning_rate=1.0,
                                            n_estimators=100))]
        else:
            self.clfs = classifiers
        
    def removeLowVarianceEntities(self,X, varianceThreshold= VarianceThreshold(threshold=0)):
        """
        This function remove low variance entities.

        Attributes:
            X: an array-like of shape (n-samples,n-features). This array represent the input samples to select the features and remove the low variance features.
            varianceThreshold: the varianceThreshold function. By defect it is used VarianceThreshold(threshold=0)
        Returns:
            A pandas DataFrame with selectedFeatures
        """
        sel = varianceThreshold
        arr = sel.fit_transform(X)
        df = pd.DataFrame(arr, columns=sel.get_feature_names_out(X.columns.values))
        display(df)
        return df

    def univariateFeatureSelection(self,X,y, k=10):
        """
        A function that select features according to the k highest scores.

        Attributes:
            X: array-like of shape (n_samples, n_features). The training input samples.
            y: array-like of shape (n_samples,)
            k: an integer that represent the number of top features to select. By defect, we use 10. 
        Returns:
            A tuple form by kbest_classif_columns and kbest_mi_columns
            kbest_classif_columns: a list with the selected characteristics by analysis of variance (ANOVA). 
                                   ANOVA measures the difference between the means of several classes.
            kbest_mi_columns: a list with the selected features by the mutual information selection method between features and labels.
                              Mutual information measures the dependence between two variables and selects features according to their ability to provide information about the target variable.
        """
        kbest_classif = SelectKBest(f_classif, k=k) # Elimina todo menos las k características de puntuación más alta
        kbest_classif.fit(X, y)
        kbest_classif_columns = kbest_classif.get_feature_names_out(X.columns.values)
        print("KBest f_classif", kbest_classif_columns)

        kbest_mi = SelectKBest(mutual_info_classif, k=k)
        kbest_mi.fit(X, y)
        kbest_mi_columns = kbest_mi.get_feature_names_out(X.columns.values)
        print("KBest Mutual Information", kbest_mi_columns)
        return kbest_classif_columns, kbest_mi_columns
    
    def featureSelectionSelectFromModel(self,X,y, estimator= LinearSVC(C=0.01, penalty="l1", dual=False)):
        """
        A function that use a meta-transformer for selecting features based on importance weights.

        Attributes:
            X: array-like of shape (n_samples, n_features). The training input samples.
            y: array-like of shape (n_samples,)
            model: The base estimator from which the transformer is built. This must be non-fitted estimator because inside of the function we are going to fit it. 
                   The estimator should have a fit method and feature_importances_ or coef_ attribute after fitting. By defect we use a LinearSVC(C=0.01, penalty="l1", dual=False)
        Return:
            A pandas DataFrame with the result of apply the estimator defined in the attributes of a function. 
            This DataFrame only contain the selected features by the classifier.
            
        """
        lsvcFitted = estimator.fit(X, y)
        model = SelectFromModel(lsvcFitted, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)
        print(X.columns[model.get_support(indices=True)])
        display(pd.DataFrame(X_new, columns=X.columns[model.get_support(indices=True)]))
        return X_new
    
    def sequentialFeatureSelection(self,X,y, num_features,estimator= RidgeCV(alphas=np.logspace(-6, 6, num=5))):
        """
        A function that performs Sequential Feature Selection..

        Attributes:
            X: array-like of shape (n_samples, n_features). The training input samples.
            y: array-like of shape (n_samples,)
            num_features: an integer with the number of features that were selected,
            estimator: an unfitted estimator
        Return:
            A tuple form by sfs_forward and sfs_backward.
            sfs_forward: Object that safe information about the selected features during the forward selection features.
                         This object include information about the selection features, the score associated to these metrics, etc.
            sfs_backward: Object that safe information about the selected features during the backward selection features.
                         This object include information about the selection features, the score associated to these metrics, etc.
        """
        ridge = estimator.fit(X,range(0,len(y))) # Errores
        tic_fwd = time()
        sfs_forward = SequentialFeatureSelector(
            ridge, n_features_to_select=num_features, direction="forward"
        ).fit(X, y)
        toc_fwd = time()

        tic_bwd = time()
        sfs_backward = SequentialFeatureSelector(
            ridge, n_features_to_select=num_features, direction="backward"
        ).fit(X, y)
        toc_bwd = time()
        
        return sfs_forward, sfs_backward
    
    def eval_classifiers(self, X, y,cv=5):
        """
        A function that apply different classifiers to the entries (X and y) using cross validation.

        Attributes:
            X: array-like of shape (n_samples, n_features). The training input samples.
            y: array-like of shape (n_samples,)
            cv: integer that determines the cross-validation splitting strategy. By defect we use 5.
        Returns:
            A pandas DataFrame that contains the average performance metrics ('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted') and standard deviation for each classifier after cross-validation. 
            This DataFrame is sorted by f-score (Mean). The rows of the Dataframe are the name of the different classifiers that are used.
        """
        # Vamos devolver los resultados como una tabla
        # Cada fila un algoritmo, cada columna un resultado
        results = pd.DataFrame(columns=['accuracy (Mean)', 'precision (Mean)', 'recall (Mean)', 'f-score (Mean)', 'accuracy (Std)', 'precision (Std)', 'recall (Std)', 'f-score (Std)'])
        
        for alg, clf in  self.clfs:
            scores = cross_validate(clf, X, y, cv=cv,
                                    scoring=('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted')) # leave-one-out cross validation
            results.loc[alg,:] = [np.mean(scores['test_accuracy']),
                                np.mean(scores['test_precision_weighted']),
                                np.mean(scores['test_recall_weighted']),
                                np.mean(scores['test_f1_weighted']),np.std(scores['test_accuracy']),
                                np.std(scores['test_precision_weighted']),
                                np.std(scores['test_recall_weighted']),
                                np.std(scores['test_f1_weighted'])]
        return results.sort_values(by='f-score (Mean)', ascending=False)
    

    def classificationBoW(self,tdf, text_column, column_to_apply, label2id, group_by, calculate_for_all_group=True):
        """
        A function that classify the text using Bag of Words

        Attributes:
            tdf: the Pandas DataFrame that contains the data to classify 
            text_column: an string with the namme of the text column where we want to apply a BoW classification 
            column_to_apply: an string with the name of the column where label2id is going to apply,
            label2id: a label2id vector
            group_by: an string with the name of the column of the DataFrame to use for the groupby. 
            calculate_for_all_group: a boolean that indicates if do you want to calculate BoW for all of the dataframe.
        Return:
            A dictionary with the results of apply different classifier to the text with the BoW features. 
            The keys of the dictionary are each group of the group_by and 'all' in case that calculate_for_all_group==True.
            The values are the pandas DataFrame returned by eval_classifier method using BoW characteristics calculated.  
        """
        vectorizer = TfidfVectorizer()
        groupValues = list(tdf[group_by].unique())
        allResults = {}
        if calculate_for_all_group:
            stdf = tdf.dropna()
            X = vectorizer.fit_transform(stdf[text_column])
            y = stdf[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
            allResults['all']=self.eval_classifiers(X, y)
        for group in groupValues:
            stdf = tdf[tdf[group_by] == group].dropna()
            X = vectorizer.fit_transform(stdf[text_column])
            y = stdf[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
            allResults[group]=self.eval_classifiers(X, y)
        return allResults

    def classificationWithFeatures(self, ext_df, ext_numeric_cols, column_to_apply, label2id, group_by= None, calculate_for_all_group = True):
        """
        A function that classify the text using aditional features calculated with the textflow analyzers.

        Attributes:
            ext_df: the Pandas DataFrame that contains the data to classify 
            ext_numeric_cols: a list with the names of the columns of the DataFrame that are numeric.
            column_to_apply: an string with the name of the column where label2id is going to apply,
            label2id: a label2id vector
            group_by: an string with the name of the column of the DataFrame to use for the groupby. 
            calculate_for_all_group: a boolean that indicates if do you want to calculate BoW for all of the dataframe.
        Return:
            A dictionary with the results of apply different classifier to the text and the application of MinMaxScaler to the numerics features. 
            The keys of the dictionary are each group of the group_by and 'all' in case that calculate_for_all_group==True.
            The values are the pandas DataFrame returned by eval_classifier method using MinMaxScaler to the numeric features. 
        """
        warnings.filterwarnings('ignore')
        allResult = {}
        ext_df = ext_df.dropna()
        if calculate_for_all_group:
            sdf = ext_df[ext_numeric_cols]
            scaler = MinMaxScaler()
            X = scaler.fit_transform(sdf)
            y = ext_df[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
            allResult['all']=self.eval_classifiers(X, y)

        if group_by != None:
            groupValues = list(ext_df[group_by].unique())
            for group in groupValues:
                sdf = ext_df[ext_df[group_by] == group]
                scaler = MinMaxScaler()
                X = scaler.fit_transform(sdf[ext_numeric_cols])
                y = sdf[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
                allResult[group]=self.eval_classifiers(X, y)
        return allResult
    
    def classificationDeepVectors(self,ddf, label2id, column_to_apply, group_by, binary= False,calculate_for_all_group=True):
        """
        A function that classify the data using deep vectors

        Attributes:
            ddf: the Pandas DataFrame that contains the data to classify 
            label2id: a label2id vector
            column_to_apply: an string with the name of the column where label2id is going to apply,
            group_by: an string with the name of the column of the DataFrame to use for the groupby. 
            binary: a boolean value use to indicate if it is a binary classification
            calculate_for_all_group: a boolean that indicates if do you want to calculate BoW for all of the dataframe.
        Return:
            A dictionary with the results of apply different classifier to the text and deep vectors. 
            The keys of the dictionary are each group of the group_by and 'all' in case that calculate_for_all_group==True.
            The values are the pandas DataFrame returned by eval_classifier method using deep vectors. 
        """
        groupValues = list(ddf[group_by].unique())
        allResults = {}
        if calculate_for_all_group:
            sddf = ddf.copy()
            X = np.stack(sddf['encoding'])
            y = sddf[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
            allResults['all']=self.eval_classifiers(X, y)
        for group in groupValues:
            sddf = ddf[ddf[group_by] == group] if binary==False else ddf[ddf[group_by] == group].dropna()
            X = np.stack(sddf['encoding'])
            y = sddf[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
            if len(X) < 5:
                allResults[group]=self.eval_classifiers(X, y,len(X))
            else:
                allResults[group]=self.eval_classifiers(X, y)
        return allResults


    def gen_encodings(self,text,tokenizer= AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne"), model= AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")):
        """
        A function that codify a text.

        Attributes:
            text: an string of which we want to codify 
            tokenizer: the Transformer tokenizer to use. By defect the tokenizer is: AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
            model: the Transformer model to use. By defect the model is AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")
        Return:
            The first element of the last layer hidden-state. 
        """
        input = tokenizer.encode_plus(text,
                                        add_special_tokens = True,
                                        truncation = True,
                                        padding = "max_length",
                                        return_attention_mask = True,
                                        return_tensors = "pt")
        output = model(**input)
        poolerOutput=output.pooler_output.detach().numpy()[0]
        return poolerOutput
    
    def combine_all_features (self, label2id, all_df, numeric_cols,column_to_group, value, text_colummn ,column_to_apply):
        """
        A function that combine all the features for a value of an specific column of the data.

        Attributes:
            label2id: a label2id vector
            all_df: a dataframe that combine the result of the classification with Features, with BoW and with Deep Vectors
            numeric_cols: a list with the name of the numeric columns of the dataframe
            column_to_group: an string with the name of the column for which we want to filter
            value: the exact value of the column_to_group for each we want to filter
            text_colummn: an string with the name of the text column of the DataFrame.
            column_to_apply: an string with the name of the column where label2id is going to apply
        Returns:
            A pandas DataFrame with the result of apply the different classifiers to the combination of scaled numeric features, 
            TF-IDF transformed text features, and stacked encoding features for a specific group (determined by column_to_group and value).
        """
        vectorizer = TfidfVectorizer()
        sdf = all_df[all_df[column_to_group] == value].dropna()
        Xf = sdf[numeric_cols]
        scaler = MinMaxScaler()
        Xf = scaler.fit_transform(Xf)
        Xt = vectorizer.fit_transform(sdf[text_colummn]).toarray()
        Xd = np.stack(sdf['encoding'])
        y = sdf[column_to_apply].apply(lambda x: label2id[x]).to_numpy()
        print(Xf.shape, Xt.shape, Xd.shape)
        X = np.concatenate((Xf, Xt, Xd), axis=1)
        dfResult= self.eval_classifiers(X, y)
        return dfResult
    
