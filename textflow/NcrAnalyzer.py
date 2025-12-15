from typing import Optional
from textflow.Analyzer import Analyzer
import os
import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import RegexpTokenizer
import numpy as np

class NcrAnalyzer(Analyzer):
    """
    A class that provides methods to obtain Ncr features from the text of a sequence.

    Attributes:
        ncrFile: a pandas DataFrame with the Ncr lexicon.
        tokenizer: a nltk tokenizer that split the texts in words.
    """

    def __init__(self, pathFile=os.path.join(os.path.dirname(__file__), 'ncr_norm.csv'), tokenizer = RegexpTokenizer("[\w']+")):
        """
        Create a Ncr analyzer.

        Args:
            pathFile: the path of the Ncr lexicon.
            tokenizer: a nltk tokenizer that split the texts in words.
        """
        self.ncrFile = pd.read_csv(pathFile).drop(['Unnamed: 0'],axis=1)
        self.tokenizer = tokenizer

    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a polarity function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.search_terms,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def search_terms(self, arrayText):
        """
        Function that analyzes a list of texts to extract the Ncr metrics.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the results of the analysis of the correspondant text with Ncr lexicon.  The result of the analysis of each text is a list composed of 4 sublist. The 4 sublist are maximum values, the minimum values, the standard deviation values and the mean values found for each Ncr feature (anger, fear, joy, sadness).
        """
        arrayResults =[]
        for text in arrayText:
            tkn_phrase = self.tokenizer.tokenize(text.lower())
            lst_terms = self.ncrFile['Términos'].tolist()
            elms = []
            for i in tkn_phrase:
                if i in lst_terms:
                    elm = self.ncrFile.loc[self.ncrFile['Términos'] == i].values.flatten().tolist()
                    elms.append(elm)
            dir_f = {}
            for i in elms:
                dir_f[str(i[0])] = [float(x.replace(',', '.')) for x in [i[1], i[2], i[3], i[4]]]
            ar_res = []
            try:
                lst_arrays = list(dir_f.values())
                ar_max = np.max(lst_arrays, axis = 0).tolist()
                ar_min = np.min(lst_arrays, axis = 0).tolist()
                ar_std_dev = np.std(lst_arrays, axis = 0).tolist()
                ar_avg = np.mean(lst_arrays, axis = 0).tolist()
                ar_res = [ar_max, ar_min, ar_std_dev, ar_avg]
            except:
                ar_res = [[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0]]
            arrayResults.append(ar_res)
        return arrayResults

