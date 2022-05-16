import os
import spacy
import spacy.cli
from typing import Optional
from textflow.Analyzer import Analyzer
from transformers import pipeline
import torch

class EmotionAnalyzer(Analyzer):
    def __init__(self, task = "text-classification",modelEmotions = 'pysentimiento/robertuito-emotion-analysis', allScores = True):
        """Creates an analyzer from an input object.

        Args:
            function: the function of the analyzer like count word, files...
            isMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        self.emotionsClassifier = pipeline(task,model=modelEmotions, return_all_scores=allScores)


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
        super().analyze(self.emotions,sequence, tag, levelOfAnalyzer, levelOfResult, True) 


    def emotions(self, arrayText):
        arrayResults =[]
        for text in arrayText:
            prediction = self.emotionsClassifier(text)
            #arrayResults.append(prediction[0][0])
            arrayResults.append(prediction)
        return arrayResults



