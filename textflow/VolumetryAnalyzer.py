from typing import Optional
from textflow.Sequence import Sequence
from nltk.tokenize import WhitespaceTokenizer
from textflow.Analyzer import Analyzer

class VolumetryAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the volumetry of the text of a sequence.

    Attributes:
        tokenizer: the way to split the text of a sequence in tokens.
    """
    def __init__(self, tokenizer= WhitespaceTokenizer()):
        """
        Create a volumetry analyzer from an input object.
            Args:
                tokenizer: the way to split a text into token 
        """
        self.tokenizer = tokenizer


    def volumetry(self, arrayText):
        """
        Function that analyzes the volumetry of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze.
        
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for texts in arrayText:
            text = self.tokenizer.tokenize(texts)
            dicResults = { 
                "words" : len(text),
                "uniqueWords" : len(set(text)),
                "chars" : len(texts),
                "avgWordsLen" : round(len(texts) / len(text))         
            }
            arrayResults.append(dicResults)
        return arrayResults

    
    def analyze(self,sequence,tag,levelOfAnalyzer,levelOfResult:Optional[str] = ""):
        """
        Analyze a sequence with a volumetry function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.volumetry,sequence, tag, levelOfAnalyzer, levelOfResult, True)
        

   