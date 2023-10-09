from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict
from scipy.stats import shapiro, normaltest, kstest, anderson, chisquare, jarque_bera
from statsmodels.stats.diagnostic import lilliefors

class Test():
    #https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411
    def __init__(self,normalityTest=["Shapiro","D'Agostino","Anderson-Darling","Chi-Square","Lilliefors","Jarque–Bera","Kolmogorov-Smirnov"], parametricTest=["mannwhitneyu","wilcoxon","kruskal"], nonParametricTest=["mannwhitneyu","wilcoxon","kruskal"]):
        self.normalityTest = normalityTest
        self.parametricTest = parametricTest
        self.nonParametricTest = nonParametricTest
        

    def apply(self,df1,df2):
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
                    dicResult[t+' '+sig_level[i]+' sig_lev'] = list(testFinal[testFinal[t+' p-value'] > 0.05].index)
        
        return testFinal, dicResult        
                
                