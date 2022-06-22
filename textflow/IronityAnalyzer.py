from typing import Optional
from textflow.Analyzer import Analyzer
from transformers import pipeline
from transformers import AutoModelForSequenceClassification,AutoTokenizer


class IronityAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the ironity of the text of a sequence.

    Attributes:
       ironityClassifier: a pipeline that uses a model for inference the ironity of the text of a sequence. 
                          By default, the label 'NI' is non-ironic and 'I' ironic.
        maxEmbedding: The number of max_position_embedings in the config.json of the model selected.
    """

    def __init__(self, task = "text-classification",modelIronity = 'dtomas/roberta-base-bne-irony', allScores = True, maxEmbedding = 512):
        """
        Create an ironic analyzer.

        Args:
            task: the task defining which pipeline will be returned.
            model: the model that will be used by the pipeline to make predictions.
            allScores: True, if we want that the classifier returns all scores. False, in other case.
            maxEmbedding: The number of max_position_embedings in the config.json of the model selected.
        """
        if modelIronity == 'dtomas/roberta-base-bne-irony':
            model = AutoModelForSequenceClassification.from_pretrained(modelIronity)
            model.config.id2label = {0: 'NI', 1: 'I'}
            model.config.label2id = {'NI': 0, 'I': 1}
            tokenizer = AutoTokenizer.from_pretrained(modelIronity, model_max_length=512)
            self.ironityClassifier = pipeline(task,model= model, tokenizer=tokenizer,return_all_scores=allScores)
        else:
            self.ironityClassifier = pipeline(task,model= modelIronity, return_all_scores=allScores)
        self.maxEmbeding = maxEmbedding
        

    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a ironic function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.polarity,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def polarity(self, arrayText):
        """
        Function that analyzes the ironity of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for text in arrayText:
            prediction = self.ironityClassifier(text[:self.maxEmbeding])
            arrayResults.append(prediction)
        return arrayResults

