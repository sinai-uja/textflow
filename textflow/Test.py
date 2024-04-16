from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import shapiro, normaltest, kstest, anderson, chisquare, jarque_bera, mannwhitneyu, wilcoxon, kruskal, ttest_ind, ttest_rel, f_oneway 
from statsmodels.stats.diagnostic import lilliefors
from IPython.display import display, Markdown
from textflow.Visualization import Visualization

class Test():
    """
    A class that provides methods to calculate normality tests, parametric tests and non-parametric tests to the features of a DataFrame.

    Attributes:
        normalityTest: a list with the normal tests to apply
        parametricTest: a list with the parametric tests to apply
        nonParametricTest: a list with the non parametric tests to apply
        alpha: chosen significance level to interpret the p-value in parametric/non parametric tests 
    """
    #https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411
    def __init__(self,normalityTest=["Shapiro","D'Agostino","Anderson-Darling","Chi-Square","Lilliefors","Jarque–Bera","Kolmogorov-Smirnov"], parametricTest=["Students t-test","Paired Students t-Test", "ANOVA"], nonParametricTest=["mannwhitneyu","wilcoxon","kruskal"],alpha=0.05):
        """
        Initialize the Test class.

        Args:
            normalityTest: a list with the normal tests to apply
            parametricTest: a list with the parametric tests to apply
            nonParametricTest: a list with the non parametric tests to apply
            alpha: chosen significance level to interpret the p-value in parametric/non parametric tests
        """
        self.normalityTest = normalityTest
        self.parametricTest = parametricTest #https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
        self.nonParametricTest = nonParametricTest #https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
        self.alpha = alpha
        

    def report(self,df1,df2,criteriaColumn1,criteriaColumn2, visualizer = None, kdeColumn = None, boxPlotX= None, boxPlotHue=None, saveImages= False, nameFiles= None,listGraphics= ["distplot","probplot","kde","boxplot"]):
        """
        A function that apply normality test, parametric and nin-parametric tests. Moreover, it generate different graphic to visualize the data.

        Args:
            df1: the Pandas DataFrame that will used to compare in significance tests(parametric and non-parametric)
            df2: the Pandas DataFrame that will used to compare in significance tests(parametric and non-parametric)
            criteriaColumn1: an string with the value to distinct the df1 of df2
            criteriaColumn2: an string with the value to distinct the df2 of df1
            visualizer: a visualizer object of TextFlow
            kdeColumn: a string or a list with the name of the columns of the dataframe that contain the sample data from which the kde plot is created.
            boxPlotX: a list of string or a string that indicate the name of the column used to make the boxplot. If you use a list, you will generate as many boxplots as the length of this list 
            boxPlotHue: a list of string or a string that indicate the name of the column used to make the boxplot name of variable in the DataFrame. This variable represent the input for plotting long-form data
            saveImages: a booblean that indicates if we want to save the graphic.
            nameFiles= a list with the names of the files to save the graphics.
            listGraphics: a list with the name of different function to visualize the data. The name of these graphic funtions is:
                          distplot: histogram (using density normalization) and a superimposed kernel density estimated
                          probplot: probability plot
                          kde: kernel density estimate (KDE) plot
                          boxplot: box plot
                          By defect the value is ["distplot","probplot","kde","boxplot"].
        Returns:
            A dictionary that have as keys 'normalTest', 'parametricTest', 'nonParametricTest'.
            Each key have as value the result of apply the methods applyNormalTest, applyParametricTest and applyNonParametricTest
            """
        df = pd.concat([df1,df2])
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype != 'object']
        print("---------------------------------------NORMALITY TEST---------------------------------------")
        normal_results =self.applyNormalTest(df[numeric_cols])
        normal_features= set()
        for key in normal_results[1]:
            normal_features= normal_features | set(normal_results[1][key])
        print("---------------------------------------PARAMETRIC TEST---------------------------------------")
        parametricResults = self.applyParametricTest(df1, df2, criteriaColumn1,criteriaColumn2, normal_features)
        print("---------------------------------------NON-PARAMETRIC TEST---------------------------------------")    
        nonParametricResults = self.applyNonParametricTest(df1, df2, criteriaColumn1,criteriaColumn2, numeric_cols)
        dicResults = {"normalTest":normal_results,"parametricTest":parametricResults,"nonParametricTest":nonParametricResults}
        if visualizer != None:
            if "distplot" in listGraphics:
                print("---------------------------------------DISTPLOT GRAPHICS---------------------------------------")
                nf = nameFiles['displot'] if saveImages==True else None
                visualizer.show_distplots(df, numeric_cols,saveImages,nf)
            if "probplot" in listGraphics:
                print("---------------------------------------PROBPLOT GRAPHICS---------------------------------------")
                nf = nameFiles['probplot'] if saveImages==True else None
                visualizer.show_probplots(df, numeric_cols,saveImages, nf)
            if "kde" in listGraphics: #KDE COLUMN TIENE QUE SER != NONE
                print("---------------------------------------KDE GRAPHICS---------------------------------------")
                if type(kdeColumn)== list:
                    for c in kdeColumn:
                        nf = nameFiles['kde'][c] if saveImages==True else None
                        visualizer.show_kde(df, numeric_cols, kdeColumn[c],saveImages,nf)
                else:
                    nf = nameFiles['kde'] if saveImages==True else None
                    visualizer.show_kde(df, numeric_cols, kdeColumn,saveImages, nf)
            if "boxplot" in listGraphics:
                print("---------------------------------------BOXPLOT GRAPHICS---------------------------------------")
                if type(boxPlotX) == list and type(boxPlotHue) == list:
                    if len(boxPlotX)==len(boxPlotHue):
                        for x in range(len(boxPlotX)):
                            nf = nameFiles['boxplot'][x] if saveImages==True else None
                            visualizer.show_boxplot(df, numeric_cols,boxPlotX[x], boxPlotHue[x], saveImages, nf)
                elif type(boxPlotX) == list and boxPlotHue == None:
                    for x in range(len(boxPlotX)):  
                        nf = nameFiles['boxplot'][x] if saveImages==True else None 
                        visualizer.show_boxplot(df, numeric_cols,boxPlotX[x], boxPlotHue, saveImages, nf)
                else:
                    nf = nameFiles['boxplot'] if saveImages==True else None
                    visualizer.show_boxplot(df, numeric_cols,boxPlotX, boxPlotHue, saveImages, nf)
        return dicResults

    def applyNormalTest(self,df):
        """
        A function that apply normality test.

        Args:
            df: the Pandas DataFrame that will used to apply the normal tests
        Returns:
            A tuple that have at the first position a DataFrame with the results of apply the defined tests to the each numerical feature of the DataFrame.
            The second position of the tuple have a dictionary with key of the applied Test as key and as value the list of features that pass the associated test.
        """
        testFinal = pd.DataFrame()
        testFinal.index = list(df.columns)
        dicResult={}
        print("Columnas", len(df.columns))
        for i in self.normalityTest:
            if i == "Shapiro":
                try:
                  test = df.apply(lambda x: shapiro(x), axis=0)
                  test.index = ['Shapiro stat', 'Shapiro p-value']
                  test = test.transpose()
                  testFinal['Shapiro stat'] = list(test['Shapiro stat'])
                  testFinal['Shapiro p-value'] = list(test['Shapiro p-value'])
                except:
                    print("No se puede aplicar Shapiro")
                    test = df.apply(lambda x: (np.nan,np.nan), axis=0)
                    test.index = ['Shapiro stat', 'Shapiro p-value']
                    test = test.transpose()
                    testFinal['Shapiro stat'] = list(test['Shapiro stat'])
                    testFinal['Shapiro p-value'] = list(test['Shapiro p-value'])
            elif i == "D'Agostino":
                try:
                    test = df.apply(lambda x: normaltest(x), axis=0)
                    test.index = ["D'Agostino stat", "D'Agostino p-value"]
                    test = test.transpose()
                    testFinal["D'Agostino stat"] = list(test["D'Agostino stat"])
                    testFinal["D'Agostino p-value"] = list(test["D'Agostino p-value"])
                except: 
                    print("No se puede aplicar D'Agostino")
                    test = df.apply(lambda x: (np.nan,np.nan), axis=0)
                    test.index = ["D'Agostino stat", "D'Agostino p-value"]
                    test = test.transpose()
                    testFinal["D'Agostino stat"] = list(test["D'Agostino stat"])
                    testFinal["D'Agostino p-value"] = list(test["D'Agostino p-value"])

            elif i == "Anderson-Darling": 
                test = df.apply(lambda x: anderson(x), axis=0)
                test.index = ['Anderson-Darling stat', 'Anderson-Darling crit_val', 'Anderson-Darling sig_level']
                test = test.transpose()
                testFinal['Anderson-Darling stat'] = list(test['Anderson-Darling stat'])
                testFinal['Anderson-Darling crit_val'] = list(test['Anderson-Darling crit_val'])
                testFinal['Anderson-Darling sig_level'] = list(test['Anderson-Darling sig_level'])
            elif i == "Chi-Square":
                test = df.apply(lambda x: chisquare(x), axis=0)
                test.index = ['Chi-Square stat', 'Chi-Square p-value']
                test = test.transpose()
                testFinal['Chi-Square stat'] = list(test['Chi-Square stat'])
                testFinal['Chi-Square p-value'] = list(test['Chi-Square p-value'])
            elif i == "Lilliefors": 
                try:
                  test = df.apply(lambda x: lilliefors(x), axis=0)
                  test.index = ['Lilliefors stat', 'Lilliefors p-value']
                  test = test.transpose()
                  testFinal['Lilliefors stat'] = list(test['Lilliefors stat'])
                  testFinal['Lilliefors p-value'] = list(test['Lilliefors p-value'])
                except:
                  print("No se puede aplicar Lilliefors")
                  test = df.apply(lambda x: (np.nan,np.nan), axis=0)
                  test.index = ['Lilliefors stat', 'Lilliefors p-value']
                  test = test.transpose()
                  testFinal['Lilliefors stat'] = list(test['Lilliefors stat'])
                  testFinal['Lilliefors p-value'] = list(test['Lilliefors p-value'])
            elif i == "Jarque–Bera": 
                test = df.apply(lambda x: jarque_bera(x), axis=0)
                test.index = ['Jarque–Bera stat', 'Jarque–Bera p-value']
                test = test.transpose()
                testFinal['Jarque–Bera stat'] = list(test['Jarque–Bera stat'])
                testFinal['Jarque–Bera p-value'] = list(test['Jarque–Bera p-value'])
            elif i == "Kolmogorov-Smirnov":
                test = df.apply(lambda x: kstest(x, 'norm'), axis=0)
                test.index = ["Kolmogorov-Smirnov stat", "Kolmogorov-Smirnov p-value"]
                test = test.transpose()
                testFinal['Kolmogorov-Smirnov stat'] = list(test['Kolmogorov-Smirnov stat'])
                testFinal['Kolmogorov-Smirnov p-value'] = list(test['Kolmogorov-Smirnov p-value'])
        
        for t in self.normalityTest:
            if t != "Anderson-Darling":
                print("Pass the test of "+t)
                print(list(testFinal[testFinal[t+' p-value'] > 0.05].index))
                dicResult[t] = list(testFinal[testFinal[t+' p-value'] > 0.05].index)
            else:
                sig_level, crit_val = list(testFinal[t+' sig_level'])[0], list(testFinal[t+' crit_val'])[0]
                for i in range(len(crit_val)):
                    print("Pass the test of "+t)
                    print(list(testFinal[testFinal[t+' stat'] < crit_val[i]].index),"at "+str(sig_level[i])+" level of significance")
                    dicResult[t+' '+str(sig_level[i])+' sig_lev'] = list(testFinal[testFinal[t+' stat'] < crit_val[i]].index)
        
        return testFinal, dicResult        
                
    def applyNonParametricTest(self, df1, df2, criteriaColumn1,criteriaColumn2, contrastCriteriaColumns): #https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
        """
        A function that apply non parametric significant tests.

        Args:
            df1: the Pandas DataFrame that will used to compare in non-parametric significance tests
            df2: the Pandas DataFrame that will used to compare in non-parametric significance tests
            criteriaColumn1: an string with the value to distinct the df1 of df2
            criteriaColumn2: an string with the value to distinct the df1 of df2
            contrastCriteriaColumns: a list with the name of the columns to apply the significant tests
        Returns:
            A tuple that have at the first position a DataFrame with the results of apply the defined tests to the each numerical feature of the DataFrame.
            The second position of the tuple have a dictionary with key of the applied Test as key and as value a dictionary that have as key if 'Fail to reject' or 'Reject' the H0 and as value the list of features that belong to each category.
        """
        columnsDF=['Feature', 'Criteria_1', 'Criteria_2']
        dicResult = {}
        for npt in self.nonParametricTest:
            columnsDF.extend([npt+" stat", npt+" p-value"])
            dfResult = pd.DataFrame(columns=columnsDF)
            dicResult[npt] = {"Reject H0":[],"Fail to Reject H0": []} #Reject == Different distribution, Fail to Reject == Same distribution
        for col in contrastCriteriaColumns:
            row=[col, criteriaColumn1,criteriaColumn2]
            if "mannwhitneyu" in self.nonParametricTest:
                stat_mw, p_value_mw = mannwhitneyu(df1[col], df2[col])
                if p_value_mw > self.alpha:
                    dicResult["mannwhitneyu"]['Fail to Reject H0'].append(col)
                else:
                    dicResult["mannwhitneyu"]["Reject H0"].append(col)
                row.extend([stat_mw, p_value_mw])
                print(row)
            if "wilcoxon" in self.nonParametricTest:
                stat_wc = np.nan
                p_value_w = np.nan
                if len(df1) == len(df2):
                    try:
                        stat_wc, p_value_w = wilcoxon(df1[col], df2[col])
                    except:
                        print("zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements")
                    if p_value_w > self.alpha:
                        dicResult["wilcoxon"]['Fail to Reject H0'].append(col)
                    else:
                        dicResult["wilcoxon"]["Reject H0"].append(col)
                    row.extend([stat_wc, p_value_w])
                else:
                    max_len = min(len(df1), len(df2))
                    if (max_len != 0):
                      try:
                          stat_wc, p_value_w = wilcoxon(df1[col][:max_len], df2[col][:max_len])
                      except:
                          print("zero_method 'wilcox' and 'pratt' do not work if x - y is zero for all elements")
                    row.extend([stat_wc, p_value_w])

            if "kruskal" in self.nonParametricTest:
                try:
                    stat_k, p_value_k = kruskal(df1[col], df2[col])
                except:
                    stat_k, p_value_k = np.nan, np.nan
                if p_value_k > self.alpha:
                    dicResult["kruskal"]['Fail to Reject H0'].append(col)
                else:
                    dicResult["kruskal"]["Reject H0"].append(col)
                row.extend([stat_k, p_value_k])    
            print(row)
            print(dfResult)
            dfResult = dfResult._append(pd.Series(row,index=dfResult.columns), ignore_index = True)
        
        display(dfResult)
        print(dicResult)
        return dfResult, dicResult

    def applyParametricTest(self, df1, df2, criteriaColumn1,criteriaColumn2, contrastCriteriaColumns): #https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/ 
        """
        A function that apply parametric significant tests.

        Args:
            df1: the Pandas DataFrame that will used to compare in parametric significance tests
            df2: the Pandas DataFrame that will used to compare in parametric significance tests
            criteriaColumn1: an string with the value to distinct the df1 of df2
            criteriaColumn2: an string with the value to distinct the df1 of df2
            contrastCriteriaColumns: a list with the name of the columns to apply the significant tests
        Return:
            A tuple that have at the first position a DataFrame with the results of apply the defined tests to the each numerical feature of the DataFrame.
            The second position of the tuple have a dictionary with key of the applied Test as key and as value a dictionary that have as key if 'Fail to reject' or 'Reject' the H0 and as value the list of features that belong to each category.
        """
        columnsDF=['Feature', 'Criteria_1', 'Criteria_2']
        dicResult = {}
        for pt in self.parametricTest:
            columnsDF.extend([pt+" stat", pt+" p-value"])
            dfResult = pd.DataFrame(columns=columnsDF)
            dicResult[pt] = {"Reject H0":[],"Fail to Reject H0": []} #Reject == Different distribution, Fail to Reject == Same distribution
        for col in contrastCriteriaColumns:
            row=[col, criteriaColumn1,criteriaColumn2]
            if "Students t-test" in self.parametricTest:
                stat_ttestInd, p_value_ttestInd = mannwhitneyu(df1[col], df2[col])
                if p_value_ttestInd > self.alpha:
                    dicResult["Students t-test"]['Fail to Reject H0'].append(col)
                else:
                    dicResult["Students t-test"]["Reject H0"].append(col)
                row.extend([stat_ttestInd, p_value_ttestInd])
            if "Paired Students t-Test" in self.parametricTest:
                if len(df1) == len(df2):
                    stat_ttestRel, p_value_ttestRel = ttest_rel(df1[col], df2[col])
                    if p_value_ttestRel > self.alpha:
                        dicResult["Paired Students t-Test"]['Fail to Reject H0'].append(col)
                    else:
                        dicResult["Paired Students t-Test"]["Reject H0"].append(col)
                    row.extend([stat_ttestRel, p_value_ttestRel])
                else:
                    max_len = min(len(df1), len(df2))
                    if (max_len != 0):
                      try:
                          stat_ttestRel, p_value_ttestRel = ttest_rel(df1[col][:max_len], df2[col][:max_len])
                          row.extend([stat_ttestRel, p_value_ttestRel])
                      except:
                          row.extend([np.nan, np.nan])
                    
            if "ANOVA" in self.parametricTest:
                stat_anova, p_value_anova = f_oneway(df1[col], df2[col])
                if p_value_anova > self.alpha:
                    dicResult["ANOVA"]['Fail to Reject H0'].append(col)
                else:
                    dicResult["ANOVA"]["Reject H0"].append(col)
                row.extend([stat_anova, p_value_anova])    
            dfResult = dfResult._append(pd.Series(row,index=dfResult.columns), ignore_index = True)
        
        display(dfResult)
        print(dicResult)
        return dfResult, dicResult                