from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import shapiro, normaltest, kstest, anderson, chisquare, jarque_bera, mannwhitneyu, wilcoxon, kruskal, ttest_ind, ttest_rel, f_oneway 
from statsmodels.stats.diagnostic import lilliefors

class Test():
    #https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411
    def __init__(self,normalityTest=["Shapiro","D'Agostino","Anderson-Darling","Chi-Square","Lilliefors","Jarque–Bera","Kolmogorov-Smirnov"], parametricTest=["Students t-test","Paired Students t-Test", "ANOVA"], nonParametricTest=["mannwhitneyu","wilcoxon","kruskal"],alpha=0.05):
        self.normalityTest = normalityTest
        self.parametricTest = parametricTest #https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/
        self.nonParametricTest = nonParametricTest #https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/
        self.alpha = alpha
        

    def apply(self,df1,df2,criteriaColumn1,criteriaColumn2, visualizer = None):
        df = pd.concat([df1,df2], axis=1)
        numeric_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype != 'object']

        print("---------------------------------------NORMALITY TEST---------------------------------------")
        normal_results =self.applyNormalTest(df)
        normal_features= set()
        for key in normal_results[1]:
            normal_features= normal_features | set(normal_results[key])
        print("---------------------------------------PARAMETRIC TEST---------------------------------------")
        parametricResults = self.applyParametricTest(df1, df2, criteriaColumn1,criteriaColumn2, normal_features)
        print("---------------------------------------NON-PARAMETRIC TEST---------------------------------------")    
        nonParametricResults = self.applyNonParametricTest(df1, df2, criteriaColumn1,criteriaColumn2, numeric_cols)
        dicResults = {"normalTest":normal_results,"parametricTest":parametricResults,"nonParametricTes":nonParametricResults}
        return dicResults


        #Hay que poner gráficas:
        #   qUARTIL QUARTIL
        #   Box Plot
        #   Histograma
        pass

    def applyNormalTest(self,df):
        testFinal = pd.DataFrame()
        testFinal.index = list(df.columns)
        dicResult={}
        for i in self.normalityTest:
            if i == "Shapiro":
                test = df.apply(lambda x: shapiro(x), axis=0)
                test.index = ['Shapiro stat', 'Shapiro p-value']
                test = test.transpose()
                testFinal['Shapiro stat'] = list(test['Shapiro stat'])
                testFinal['Shapiro p-value'] = list(test['Shapiro p-value'])
            elif i == "D'Agostino":
                test = df.apply(lambda x: normaltest(x), axis=0)
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
                test = df.apply(lambda x: lilliefors(x), axis=0)
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
            if "wilcoxon" in self.nonParametricTest:
                if len(df1) == len(df2[col]):
                    stat_wc, p_value_w = wilcoxon(df1[col], df2[col])
                    if p_value_w > self.alpha:
                        dicResult["wilcoxon"]['Fail to Reject H0'].append(col)
                    else:
                        dicResult["wilcoxon"]["Reject H0"].append(col)
                    row.extend([stat_wc, p_value_w])
                else:
                    row.extend([np.nan, np.nan])
            if "kruskal" in self.nonParametricTest:
                stat_k, p_value_k = kruskal(df1[col], df2[col])
                if p_value_k > self.alpha:
                    dicResult["kruskal"]['Fail to Reject H0'].append(col)
                else:
                    dicResult["kruskal"]["Reject H0"].append(col)
                row.extend([stat_k, p_value_k])    
            dfResult = dfResult._append(pd.Series(row,index=dfResult.columns), ignore_index = True)
        
        print(dfResult)
        print(dicResult)
        return dfResult, dicResult

    def applyParametricTest(self, df1, df2, criteriaColumn1,criteriaColumn2, contrastCriteriaColumns): #https://machinelearningmastery.com/parametric-statistical-significance-tests-in-python/ 
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
                if len(df1) == len(df2[col]):
                    stat_ttestRel, p_value_ttestRel = ttest_rel(df1[col], df2[col])
                    if p_value_ttestRel > self.alpha:
                        dicResult["Paired Students t-Test"]['Fail to Reject H0'].append(col)
                    else:
                        dicResult["Paired Students t-Test"]["Reject H0"].append(col)
                    row.extend([stat_ttestRel, p_value_ttestRel])
                else:
                    row.extend([np.nan, np.nan])
            if "ANOVA" in self.parametricTest:
                stat_anova, p_value_anova = f_oneway(df1[col], df2[col])
                if p_value_anova > self.alpha:
                    dicResult["ANOVA"]['Fail to Reject H0'].append(col)
                else:
                    dicResult["ANOVA"]["Reject H0"].append(col)
                row.extend([stat_anova, p_value_anova])    
            dfResult = dfResult._append(pd.Series(row,index=dfResult.columns), ignore_index = True)
        
        print(dfResult)
        print(dicResult)
        return dfResult, dicResult                