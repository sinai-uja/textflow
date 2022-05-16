
import spacy
import spacy.cli
from typing import Optional
from textflow.Analyzer import Analyzer

spacy.cli.download("es_core_news_sm")

class LemmaAnalyzer(Analyzer):
    def __init__(self, nlp = spacy.load("es_core_news_sm"), posNoContent = ["PUNCT", "SPACE", "SYM"]):
        """Creates an analyzer from an input object.

        Args:
            function: the function of the analyzer like count word, files...
            isMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        self.nlp = nlp
        self.posNoContent = posNoContent

    #Este analizador, solo puede analizar cadenas de texto, por lo que solo tiene sentido que use el atributo text de metadata
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): #TODO
        """Analyze a sequence

        Args:
            sequence: the Sequence we want to analyze
            tag: the label to store the analysis resut
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result(la subsequencia a analizar dentro de la sequencia en la que queremos almacenar el resultado)
            levelOfResult: the path of the sequence level to store the result. (Podemos querer analizar los tokens pero almacenarlo a nivel de oracion)
            analyzeMetadata: boolean, if the result of the analyzer is applied in metadata (True) or in children(False)

        Raises:
            ValueError if the levelOfResult is incorrect
        """
        super().analyze(self.lemmas,sequence, tag, levelOfAnalyzer, levelOfResult, True)


    def lemmas(self, arrayText):
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

            
                                
                                



