from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict
from scipy.stats import shapiro, normaltest, kstest, anderson, chisquare, jarque_bera
from statsmodels.stats.diagnostic import lilliefors

class Test():
    #https://towardsdatascience.com/normality-tests-in-python-31e04aa4f411
    def __init__(self,parametricTest=["Shapiro","D'Agostino","Anderson-Darling","Chi-Square","Lilliefors","Jarque–Bera","Kolmogorov-Smirnov"],nonParametricTest=["mannwhitneyu","wilcoxon","kruskal"]):
        self.parametricTest = parametricTest
        self.nonParametricTest = nonParametricTest
        

    def apply(self,df1,df2):
        #Hay que poner gráficas:
        #   qUARTIL QUARTIL
        #   Box Plot
        #   Histograma
        pass

    def applyParametric(self,df):
        testFinal = pd.DataFrame()
        #Numeric Cols, hay que filtrar el df
        for i in self.parametricTest:
            if i == "Shapiro":
                test = df.apply(lambda x: shapiro(x), axis=0)
                test.index = ['Shapiro stat', 'Shapiro p-value']
                test = test.transpose()
            elif i == "D'Agostino":
                test = df.apply(lambda x: normaltest(x), axis=0)
                test.index = ["D'Agostino stat", "D'Agostino p-value"]
                test = test.transpose()
            elif i == "Anderson-Darling": 
                test = df.apply(lambda x: anderson(x), axis=0)
                test.index = ['Anderson-Darling stat', 'Anderson-Darling crit_val', 'Anderson-Darling sig_level']
                test = test.transpose()
                pass
            elif i == "Chi-Square":
                test = df.apply(lambda x: chisquare(x), axis=0)
                test.index = ['Chi-Square stat', 'Chi-Square p-value']
                test = test.transpose()
                pass
            elif i == "Lilliefors": 
                test = df.apply(lambda x: lilliefors(x), axis=0)
                test.index = ['Lilliefors stat', 'Lilliefors p-value']
                test = test.transpose()
                pass
            elif i == "Jarque–Bera": 
                test = df.apply(lambda x: jarque_bera(x), axis=0)
                test.index = ['Shapiro stat', 'Shapiro p-value']
                test = test.transpose()
                pass
            elif i == "Kolmogorov-Smirnov":
                test = df.apply(lambda x: kstest(x, 'norm'), axis=0)
                test.index = ["Kolmogorov-Smirnov stat", "Kolmogorov-Smirnov p-value"]
                test = test.transpose()
        
        for t in self.parametricTest:
            if t != "Anderson-Darling":
                print("Pass the test of"+t)
                print(list(test[test[t+' p-value'] > 0.05].index))
            else:
                for i in range(len(list(test[t+' crit_val'].index))):
                    sig_level, crit_val = test[t+' sig_level'][i], test[t+' crit_val'][i]
                    print("Pass the test of"+t)
                    print(list(test[test[t+' stat'] < crit_val].index),"at {sig_level} level of significance")
                
                
                