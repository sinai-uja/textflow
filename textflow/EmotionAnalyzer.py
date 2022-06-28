import os
import spacy
import spacy.cli
from typing import Optional
from textflow.Analyzer import Analyzer
from transformers import pipeline

class EmotionAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the emotions of the text of a sequence.

    Attributes:
        polarityClassifier: a pipeline that uses a model for inference the emotions of the text of a sequence.
    """

    def __init__(self, task = "text-classification",modelEmotions = 'pysentimiento/robertuito-emotion-analysis', allScores = True, maxEmbedding = 130):
        """
        Create a emotions analyzer.

        Args:
            task: the task defining which pipeline will be returned.
            model: the model that will be used by the pipeline to make predictions.
            allScores: True, if we want that the classifier returns all scores. False, in other case.
            maxEmbedding: The number of max_position_embeddings in the config.json of the model selected.
        """
        self.emotionsClassifier = pipeline(task,model=modelEmotions, return_all_scores=allScores)
        self.maxEmbedding = maxEmbedding


    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a emotion function.
            
        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result. 
        """
        super().analyze(self.emotions,sequence, tag, levelOfAnalyzer, levelOfResult, True) 


    def emotions(self, arrayText):
        """
        Function that analyzes the emotions of a list of texts.
            
        Args:
            arrayText: list that contains the texts that we want to analyze.
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for text in arrayText:
            prediction = self.emotionsClassifier(text[:self.maxEmbedding])
            #arrayResults.append(prediction[0][0])
            arrayResults.append(prediction)
        return arrayResults



