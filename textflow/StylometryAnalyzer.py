import string
from typing import Optional

#import spacy
#import spacy.cli
from nltk.text import Text
from nltk.tokenize import WhitespaceTokenizer
import math

from textflow.Analyzer import Analyzer

class StylometryAnalyzer(Analyzer): #TODO

    def __init__(self,stopwords, puntuation = string.punctuation,tokenizer = WhitespaceTokenizer()):
        self.stopwords = stopwords
        self.puntuation = puntuation
        self.tokenizer = tokenizer

    #Este analizador, solo puede analizar cadenas de texto, por lo que solo tiene sentido que use el atributo text de metadata
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str]= ""):
        super().analyze(self.stylometry,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def stylometry(self, arrayText):
        resultsList = []
        for t in arrayText:
            #doc = self.nlp(text)
            t.lower()
            tokens = self.tokenizer.tokenize (t)
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
        self.uniqueWords = [token[0] for token in self.freqWord]
        self.numWordFreqOne = len( [token[0] for token in self.freqWord if token[1] == 1 ])
        self.TTR = len(self.uniqueWords) / len(text)
        self.RTTR = len(self.uniqueWords) / math.sqrt(len(text))
        self.herdan = math.log(len(self.uniqueWords),10) / math.log(len(text),10)
        self.mass = (math.log(len(text),10)- math.log(len(self.uniqueWords),10)) /  pow(math.log(len(self.uniqueWords),10),2)
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