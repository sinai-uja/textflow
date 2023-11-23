from typing import Optional
from textflow.Analyzer import Analyzer
from nltk.tokenize import WhitespaceTokenizer
from nltk.corpus import stopwords
import sklearn.feature_extraction.text
from nltk import ngrams

class NGramsAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the n-grams of the text of a sequence.

    Attributes:
        stopwords: a list with stopwords.
        tokenizer: a function to tokenize the text.
        ngramsSize: a number with the size of the n-grams.
        listOfNGrams: a list witn the n-grams of the text to analyze.
        freqNGrams: a dictionary with the different n-grams and their frequence in the text to analyze.
    """

    def __init__(self, stopwords=stopwords.words('spanish'), tokenizer = WhitespaceTokenizer(), ngramsSize = 2):
        """
        Create a n-grams analyzer.

        Args:
            stopwords: a list with stopwords.
            tokenizer: a function to tokenize the text.
            ngramsSize: a number with the size of the n-grams.
        """
        self.stopwords = stopwords
        self.tokenizer = tokenizer
        self.ngramsSize = ngramsSize
        

    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a n-grams function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result.
        """
        super().analyze(self.ngrams,sequence, tag, levelOfAnalyzer, levelOfResult, True)

    def ngrams(self, arrayText):
        """
        Function that analyzes the n-grams of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for text in arrayText:
            self.countFreqNGrams(text)
            prediction = {
                'n-grams': self.listOfNGrams, #unique n-grams
                'freqN-Grams': self.freqNGrams
            }
            arrayResults.append(prediction)
        return arrayResults

    def countFreqNGrams(self,textOriginal):
        """
        Function that divide the text in n-grams, and count the frequence of them.
    
        Args:
            text: a string/text to analyze
        """
        try:
            vect = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(self.ngramsSize,self.ngramsSize),tokenizer=self.tokenizer.tokenize,stop_words= self.stopwords)
            text=[textOriginal]
            vect.fit(text)
            self.listOfNGrams = vect.get_feature_names_out().tolist()
            dicfreq={}

            ngramsText= [token.lower() for token in self.tokenizer.tokenize(textOriginal) if token.lower() not in self.stopwords]
            for i in ngrams(ngramsText, self.ngramsSize):
                if ' '.join(i) in dicfreq:
                    dicfreq[' '.join(i)] += 1
                else:
                    dicfreq[' '.join(i)] = 1
            self.freqNGrams = dicfreq
        except Exception:
            self.listOfNGrams = []
            self.freqNGrams = {}

