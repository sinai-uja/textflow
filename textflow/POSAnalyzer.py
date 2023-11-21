import os
import spacy
import spacy.cli
from typing import Optional
from textflow.Analyzer import Analyzer

spacy.cli.download("es_core_news_sm")

class POSAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the part-of-speech(POS) of the text of a sequence.

    Attributes:
        nlp: a model of language.
    """

    def __init__(self, nlp = spacy.load("es_core_news_sm")):
        """
        Create a POS analyzer from an input object.

        Args:
            nlp: a model of language.
        """
        self.nlp = nlp

    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a POS Tagger function.
        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result. 
        """
        super().analyze(self.pos,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def pos(self,arrayText):
        '''
        Function that get the POS tag of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze.
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        '''
        arrayResults = []
        for text in arrayText:
            srcPOS = []
            srcPOSTag = []
            dicFreqPOS = {}
            dicRelFreqPOS = {}
            dicPOSTokens = {}
            doc = self.nlp(text)
            for token in doc:
                srcPOS.append(token.pos_)
                srcPOSTag.append((str(token).lower(),token.pos_))
                dicPOSTokens[token.pos_] = dicPOSTokens.get(token.pos_, {})
                if token.pos_ in dicFreqPOS:
                    dicPOSTokens[token.pos_][token.text] = dicPOSTokens[token.pos_].get(token.text, 0) + 1
                    dicFreqPOS[token.pos_] += 1
                else:
                    dicFreqPOS[token.pos_] = 1
                    dicPOSTokens[token.pos_][token.text] = dicPOSTokens[token.pos_].get(token.text, 0) + 1
            for tag in dicFreqPOS:
                dicRelFreqPOS[tag] = dicFreqPOS[tag] / len(doc)
            pos = {
                "srcPOS": srcPOS,
                "srcPOSTag": srcPOSTag,
                "FreqPOS": dicFreqPOS,
                "RelFreqPOS": dicRelFreqPOS,
                "POSTokens": dicPOSTokens
            }
            arrayResults.append(pos)
        return arrayResults    
                                


