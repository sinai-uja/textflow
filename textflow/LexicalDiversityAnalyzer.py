import string
from typing import Optional

from nltk.text import Text
from nltk.tokenize import WhitespaceTokenizer
from lexical_diversity import lex_div as ld
import math

from textflow.Analyzer import Analyzer

class LexicalDiversityAnalyzer(Analyzer): 
    """
    A class that provides methods to analyze the lexical diversity of the text of a sequence.

    Attributes:
        lemmatizer: a function that tokenize the text in lemmas (preferred than original words)
    """
    def __init__(self, lemmatizer):
        """
        Create a stylometry analyzer from an input object.

        Args:
            lemmatizer: a function to tokenize the text in lemmas
        """
        self.lemmatizer = lemmatizer
    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str]= ""):
        """
        Analyze a sequence with a lexical a diversity function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.lexicalDiversity, sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def lexicalDiversity(self, arrayText):
        '''
        Function that get the lexical diversity measures  of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        '''
        resultsList = []
        for t in arrayText:
            lemmas = self.lemmatizer(t.lower())
            result={
                "SimpleTTR": ld.ttr(lemmas),
                "RootTTR": ld.root_ttr(lemmas),
                "LogTTR": ld.log_ttr(lemmas),
                "MaasTTR": ld.maas_ttr(lemmas),
                "MSTTR": ld.msttr(lemmas),
                "MATTR": ld.mattr(lemmas),
                "HDD": ld.hdd(lemmas),
                "MTLD": ld.mtld(lemmas),
                "MTLDMAWrap": ld.mtld_ma_wrap(lemmas),
                "MTLDMABi": ld.mtld_ma_bid(lemmas)
            }
            resultsList.append(result)
        return resultsList