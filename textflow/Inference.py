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


class Inference():
    
    def __init__(self, classifiers= None):
        if classifiers == None:
            self.clfs = [('Logistic regression', LogisticRegression(max_iter=1000)),
                ('SVM', SVC()),
                ('Decision tree', DecisionTreeClassifier()),
                ('RandomForest', RandomForestClassifier(n_estimators=20, random_state=45)),
                ('SGD', SGDClassifier(max_iter=1000, tol=1e-4, random_state=45)),
                ('KNN', KNeighborsClassifier()),
                ('MLP', MLPClassifier(max_iter=1000))
            ]
        else:
            self.clfs = classifiers
        
    def removeLowVarianceEntities(self,X, varianceThreshold= VarianceThreshold(threshold=0)):
        sel = varianceThreshold
        arr = sel.fit_transform(X)
        df = pd.DataFrame(arr, columns=sel.get_feature_names_out(X.columns.values))
        display(df)
        return df

    def univariateFeatureSelection(self,X,y, k=10):
        kbest_classif = SelectKBest(f_classif, k=k) # Elimina todo menos las k características de puntuación más alta
        kbest_classif.fit(X, y)
        kbest_classif_columns = kbest_classif.get_feature_names_out(X.columns.values)
        print("KBest f_classif", kbest_classif_columns)

        kbest_mi = SelectKBest(mutual_info_classif, k=k)
        kbest_mi.fit(X, y)
        kbest_mi_columns = kbest_mi.get_feature_names_out(X.columns.values)
        print("KBest Mutual Information", kbest_mi_columns)
        return kbest_classif_columns, kbest_mi_columns
    
    def featureSelectionSelectFromModel(self,X,y, lsvc= LinearSVC(C=0.01, penalty="l1", dual=False)):
        
        lsvcFitted = lsvc.fit(X, y)
        model = SelectFromModel(lsvcFitted, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)
        print(X.columns[model.get_support(indices=True)])
        display(pd.DataFrame(X_new, columns=X.columns[model.get_support(indices=True)]))
        return X_new
    
    def sequentialFeatureSelection(self,X,y, num_features,ridgecv= RidgeCV(alphas=np.logspace(-6, 6, num=5))):
        ridge = ridgecv.fit(X,range(0,len(y))) # Errores
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

        # Vamos devolver los resultados como una tabla
        # Cada fila un algoritmo, cada columna un resultado
        results = pd.DataFrame(columns=['accuracy', 'precision', 'recall', 'f-score'])
        for alg, clf in  self.clfs:
            scores = cross_validate(clf, X, y, cv=cv,
                                    scoring=('accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted')) # leave-one-out cross validation
            results.loc[alg,:] = [np.mean(scores['test_accuracy']),
                                np.mean(scores['test_precision_weighted']),
                                np.mean(scores['test_recall_weighted']),
                                np.mean(scores['test_f1_weighted'])]
        return results.sort_values(by='f-score', ascending=False)
    

    def classificationBoW(self,tdf, text_column, column_to_apply, label2id, group_by, calculate_for_all_group=True):
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
    
    def classificationDeepVectors(self,ddf, label2id, column_to_apply, text_column, group_by, binary= False,calculate_for_all_group=True):
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
            allResults[group]=self.eval_classifiers(X, y)
        return allResults


    def gen_encodings(self,text,tokenizer= AutoTokenizer.from_pretrained("PlanTL-GOB-ES/roberta-base-bne"), model= AutoModel.from_pretrained("PlanTL-GOB-ES/roberta-base-bne")):
        input = tokenizer.encode_plus(text,
                                        add_special_tokens = True,
                                        truncation = True,
                                        padding = "max_length",
                                        return_attention_mask = True,
                                        return_tensors = "pt")
        output = model(**input)
        return output.pooler_output.detach().numpy()[0]
    
    def combine_all_features (self, label2id, all_df, numeric_cols,column_to_group, value, text_colummn ,column_to_apply):
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
        return self.eval_classifiers(X, y)
    
