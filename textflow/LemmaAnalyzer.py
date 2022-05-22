
import spacy
import spacy.cli
from typing import Optional
from textflow.Analyzer import Analyzer

spacy.cli.download("es_core_news_sm")

class LemmaAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the lemmas of the text of a sequence.

    Attributes:
        nlp: a model of language.
        posNoContent: a list with the POS tag from which we don't want to get the lemma. 
                      Some of the POS tag that haven't content are "PUNCT", "SPACE", "SYM".
    """

    def __init__(self, nlp = spacy.load("es_core_news_sm"), posNoContent = ["PUNCT", "SPACE", "SYM"]):
        """Create an analyzer from an input object.

        Args:
            nlp: a model of language.
            posNoContent: a list with the POS tag from which we don't want to get the lemma.
        """
        self.nlp = nlp
        self.posNoContent = posNoContent

    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a lemma function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result. 
        """
        super().analyze(self.lemmas,sequence, tag, levelOfAnalyzer, levelOfResult, True)


    def lemmas(self, arrayText):
        '''
        Function that get the lemmas of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        '''
        arrayResult = []
        for text in arrayText:
            sequenceLemmas = []
            setLemmas = set()
            sumaLenLemmas=0
            doc= self.nlp(text)
            for token in doc:
                if token.pos_ not in self.posNoContent:
                    sumaLenLemmas += len(token.lemma_)
                    setLemmas.add(token.lemma_)
                    sequenceLemmas.append(token.lemma_)
            lemma={
                "srclemmas" : sequenceLemmas,
                "uniqueLemmas" : len(setLemmas),
                "avgLemmas" : round(sumaLenLemmas/len(sequenceLemmas)) 
            }
            arrayResult.append(lemma)
        return arrayResult

            
                                
                                



