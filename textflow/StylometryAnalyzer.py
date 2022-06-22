import string
from typing import Optional

from nltk.text import Text
from nltk.tokenize import WhitespaceTokenizer
import math

from textflow.Analyzer import Analyzer

class StylometryAnalyzer(Analyzer): 
    """
    A class that provides methods to analyze the stylometry of the text of a sequence.

    Attributes:
        stopwords: a list with stopwords.
        puntuation: a list with puntuationMarks.
        tokenizer: a function to tokenize the text.
        uniqueWords: a list with the vocabulary of a text.
        numWordFreqOne: the numbers of words that only appear one time in the text. 
        TTR: type-token ratio.
        RTTR: root type-token ratio.
        herdan: the index of Herdan.
        mass: the index of Mass.
        somers: the index of Somers.
        dugast: the index of Dugast.
        honore: the index of Honoré.
        freqStopWords: the frequence of the stopwords in the text.
        freqPuntuationMarks: the frequence of the different puntuations marks in the text.
        freqWord: the frequence of the different words in the text.
    """
    def __init__(self,stopwords, puntuation = string.punctuation,tokenizer = WhitespaceTokenizer()):
        """
        Create a stylometry analyzer from an input object.

        Args:
            stopwords: a list with stopwords
            puntuation: a list with puntuationMarks
            tokenizer: a function to tokenize the text
        """
        self.stopwords = stopwords
        self.puntuation = puntuation
        self.tokenizer = tokenizer

    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str]= ""):
        """
        Analyze a sequence with a stylometry function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.stylometry,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def stylometry(self, arrayText):
        '''
        Function that get the stylometry (somes index, frequence of words ) of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        '''
        resultsList = []
        for t in arrayText:
            t.lower()
            tokens = self.tokenizer.tokenize(t)
            text= [token.lower() for token in tokens]
            self.freqWords(text,self.stopwords,self.puntuation)
            self.funcionesTTR(text)
            result={
                "uniqueWords": len(self.uniqueWords),
                "TTR": self.TTR,
                "RTTR": self.RTTR,
                "Herdan": self.herdan,
                "Mass": self.mass,
                "Somers": self.somers,
                "Dugast": self.dugast,
                "Honore": self.honore,
                "FreqStopWords": self.freqStopWords,
                "FreqPuntuationMarks": self.freqPuntuationMarks,
                "FreqWords": self.freqWord
            }
            resultsList.append(result)
        return resultsList

    def funcionesTTR(self, text):
        """
        Function that calculate different TTR index.

        Args:
            text: a string with the text to analyze.
        """
        self.uniqueWords = [token[0] for token in self.freqWord]
        self.numWordFreqOne = len( [token[0] for token in self.freqWord if token[1] == 1 ])
        self.TTR = len(self.uniqueWords) / len(text)
        self.RTTR = len(self.uniqueWords) / math.sqrt(len(text))
        if len(text)== 1:
            self.herdan = math.log(len(self.uniqueWords),10)
        else:
            self.herdan = math.log(len(self.uniqueWords),10) / math.log(len(text),10)
        if pow(math.log(len(self.uniqueWords),10),2) == 0:
            self.mass = (math.log(len(text),10)- math.log(len(self.uniqueWords),10))
        else:
            self.mass = (math.log(len(text),10)- math.log(len(self.uniqueWords),10)) /  pow(math.log(len(self.uniqueWords),10),2)
        if len(text) == 10:
            self.somers = math.log(math.log(len(self.uniqueWords),10),10)
        elif len(self.uniqueWords) == 10 or len(self.uniqueWords) == 1:
            self.somers = 0
        else:
            self.somers = math.log(math.log(len(self.uniqueWords),10),10) / math.log(math.log(len(text),10),10)
        if math.log(len(text),10)- math.log(len(self.uniqueWords),10) == 0:
            self.dugast = pow(math.log(len(text),10),2)
        else:
            self.dugast = pow(math.log(len(text),10),2) / (math.log(len(text),10)- math.log(len(self.uniqueWords),10))
        if 1-(self.numWordFreqOne/len(self.uniqueWords)) == 0:
            self.honore = 100*(math.log(len(text),10))
        else:
            self.honore = 100*(math.log(len(text),10)/(1-(self.numWordFreqOne/len(self.uniqueWords))))    


    def freqWords(self,tokens, stopWords, puntuationMarks):
        """
        Function that count the frequence of stopWords, puntuationMarks and words of a list of tokens.

        Args:
            tokens: a list of tokens that we want to count the frequence.
            stopwords: a list with the stopwords.
            puntuationMarks: a list with the puntuation marks.
        """
        freqStopWords = {}
        freqPuntuationMarks = {}
        freqWord ={} 
        for token in tokens:
            if token in stopWords:
                if token in freqStopWords:
                    freqStopWords[token] += 1
                else:
                    freqStopWords[token] = 1
            elif token in puntuationMarks:
                if token in freqPuntuationMarks:
                    freqPuntuationMarks[token] += 1
                else:
                    freqPuntuationMarks[token] = 1
            else: 
                if token in freqWord:
                    freqWord[token] += 1
                else:
                    freqWord[token] = 1
        self.freqWord = sorted(freqWord.items(), reverse = True)
        self.freqPuntuationMarks = sorted(freqPuntuationMarks.items(), reverse = True)
        self.freqStopWords = sorted(freqStopWords.items(), reverse = True)   