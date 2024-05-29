import os
import spacy
import spacy.cli
from typing import Optional
from textflow.Sequence import Sequence
import re
import numpy as np
import math
from functools import reduce
from textflow.Analyzer import Analyzer


creaPath = os.path.join(os.path.dirname(__file__), 'Crea-5000.txt')
spacy.cli.download("es_core_news_sm")

class ComplexityAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the complexity of the text of a sequence.

    Attributes:
        nlp: a model of language.
        dicFreqWords: a dictionary with the most frequence words of spanish language.
        numContentSentences: the number of the content sentences in the text.
        numComplexSents: the nomber of complex sentences in the text.
        avgLenSentence: the average length of the sentences in the text. 
        numPuntuationMark: the number of punctuation marks in the text.
        numWords: the number of words in the text.
        numRareWord: the number of rare words in the text.
        numSyllabes: the number of syllabes in the text.
        numChars: the number of chars in the text.
        indexLowFreqWords: the index of low frequence words of a text.           
        lexicalDistributionIndex: the index of lexical distribution of a text.               
        lexicalComplexity: the index of lexical complexity of a text.
        spauldingScore: the Spaulding's readability score.
        sentenceComplexityIndex: the index of sentence complexity.
        autoReadabilityIndex: the autoreadability index of a text.                  
        readabilityFH: the Fernandez Huerta readability of a text.
        perspicuityIFSZ: the Flesch-Szigriszt perspicuity of a text. 
        poliniComprensibility: the Polini comprehensibility of a text.
        muLegibility: the Mu legibility of a text.
        minAge: the minimum age to read a text. 
        solReadability: the SOL readability of a text.
        crawford: the Crawford's years of a text.
        min_max_list: the minimum of maximum tree depths.
        max_max_list: the maximum of maximum tree depths.
        mean_max_list: the mean of maximum tree depths.
    """


    def __init__(self, rutaArchivoCrea = creaPath, nlp = spacy.load("es_core_news_sm")):
        """
        Create a complexity analyzer from an input object.

        Args:
            rutaArchivoCrea: the path of the file that contains the most frequence words of spanish language
            nlp: spacy model used to calculate the analizer metrics
        """
        self.nlp = nlp
        #Vamos a cargar CREA:
        self.dicFreqWords=self.read(rutaArchivoCrea)

    
    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a complexity function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result. 
            analyzeMetadata: boolean, if the result of the analyzer is applied in metadata (True) or in children(False).

        Raises:
            ValueError if the levelOfResult is incorrect
        """
        super().analyze(self.complexity,sequence, tag, levelOfAnalyzer, levelOfResult, True)


    def read(self,fichero):
        """
        Function that read a txt File.

        Args:
            fichero: the path of the file to read.

        """
        with open(fichero,'r',encoding='latin-1') as file:
            next(file)
            lines = file.readlines()
            
        freqWordsCrea ={}    
        for l in lines[:-2]:
            words = l.strip().split()
            freqWordsCrea[words[1]] = float(words[2].replace(',',''))
        return freqWordsCrea



    def complexity(self, arrayText):
        """
        Function that analyzes the complexity of a list of texts.
            
        Args:
            arrayText: list that contains the texts that we want to analyze.

        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        """
        arrayResults =[]
        for text in arrayText:
            doc= self.nlp (text)
            self.simplesMetrics(doc)
            self.countRareAndLowWord()
            self.analyzeLegibility(doc)
            self.lexicalIndex()
            self.sentenceComplexity()
            self.readability()
            self.ageReadability()
            self.embeddingDepth()
            dicResults = {
            'nSentences' : self.numContentSentences,
            'nComplexSentence' : self.numComplexSents,
            'avglenSentence' : self.avgLenSentence, 
            'nPuntuationMarks': self.numPuntuationMark,
            'nWords': self.numWords,
            'nRareWords' : self.numRareWord,
            'nSyllabes' : self.numSyllabes,
            'nChar' : self.numChars,
            'ILFW': self.indexLowFreqWords,                  
            'LDI': self.lexicalDistributionIndex,                    
            'LC': self.lexicalComplexity,
            'SSR': self.spauldingScore,
            'SCI' : self.sentenceComplexityIndex,
            'ARI' : self.autoReadabilityIndex,                      
            'huerta': self.readabilityFH,   
            'IFSZ': self.perspicuityIFSZ,       
            'polini': self.poliniComprensibility, 
            'mu': self.muLegibility,           
            'minage': self.minAge,         
            'SOL': self.solReadability,         
            'crawford': self.crawford,
            'min_depth' : self.min_max_list,
            'max_depth' : self.max_max_list,
            'mean_depth' : self.mean_max_list              
            }
            arrayResults.append(dicResults)
        return arrayResults



    def simplesMetrics(self, doc):
        """
        Function that calculate of a doc.
            
        Args:
            doc: sequence of tokens object.
        """
        self.sentences = [s for s in doc.sents]
        self.numSentences = len(self.sentences)
        pcs = []
        for sent in self.sentences:
            docSent = self.nlp(sent.text)
            pcs.append([w for w in docSent if re.match('NOUN.*|VERB.*|ADJ.*', w.pos_) ]) 
        numPunt = 0
        numWords = 0
        numWord3Syllabes = 0
        numSyllabes = 0
        numChars = 0
        for token in doc:
            if token.pos_ == "PUNCT":
                numPunt+=1
            else:
                numWords +=1
                syllabes = self.countSyllabes(token.text)
                if syllabes > 2:
                    numWord3Syllabes +=1
                if token.text != "\r\n":
                    numSyllabes += syllabes
                    numChars = len(token.text)

        self.posContentSentences = pcs
        self.numContentSentences = len(pcs)
        self.numPuntuationMark = numPunt
        self.numWords = numWords
        self.numWords3Syllabes = numWord3Syllabes
        self.numSyllabes = numSyllabes
        self.numChars = numChars

    def analyzeLegibility(self,doc):
        """
        Function that analyze the legibility of a text.

        Args:
            doc: a sequence of tokens.
        """
        self.readabilityFH = 206.84 - 0.60*(self.numSyllabes/self.numWords) - 1.02*(self.numWords/self.numSentences)
        self.perspicuityIFSZ = 206.835 - ((62.3*self.numSyllabes)/self.numWords) - (self.numWords/self.numSentences)
        
        numLetters = 0
        listLenLetters =[]
        for token in doc:
            if token.text.isalpha(): 
                numLetters += len(token.text)
                listLenLetters.append(len(token.text))
        
        avgLettersWords = numLetters/self.numWords
        listLenLetters = np.array(listLenLetters)
        if self.numSentences == 0:
            self.poliniComprensibility = 95.2 - (9.7 * avgLettersWords) - ((0.35*self.numWords)/1)
        else:    
            self.poliniComprensibility = 95.2 - (9.7 * avgLettersWords) - ((0.35*self.numWords)/self.numSentences)        
        if self.numWords < 2:
            self.muLegibility = 0
        else:    
            self.muLegibility = (self.numWords/(self.numWords-1))*(avgLettersWords/listLenLetters.var())*100
        
    def lexicalIndex(self):
        """
        Function that calculate different lexical index of a text.
        """
        self.numContentWords = reduce((lambda a, b: a + b), [len(s) for s in self.posContentSentences])
        self.numDistinctContentWords = len(set([w.text.lower() for s in self.posContentSentences for w in s]))
        if self.numContentWords == 0:
            self.numContentWords = 1
        self.indexLowFreqWords = self.numLowWord / float(self.numContentWords)
        if self.numContentSentences == 0:
            self.numContentSentences = 1
        self.lexicalDistributionIndex = self.numDistinctContentWords / float(self.numContentSentences)
        self.lexicalComplexity = (self.indexLowFreqWords+self.lexicalDistributionIndex) /2
        

    def readability(self):
        """
        Function that calculate the readability of a text.
        """
        self.autoReadabilityIndex = 4.71 * self.numChars / self.numWords + 0.5 * self.numWords/self.numContentSentences
        self.spauldingScore = 1.609*(self.numWords / self.numContentSentences) + 331.8* (self.numRareWord /self.numWords) + 22.0

    def countRareAndLowWord(self):
        """
        Function that count the rare and low words of a text.
        """
        freqWord = sorted(self.dicFreqWords, key = self.dicFreqWords.__getitem__, reverse = True)[:1500]
        countRareWord = 0
        countLowWord = 0
        for sentence in self.posContentSentences:
            for word in sentence:
                if word.text.lower() not in freqWord:
                    countRareWord += 1
                if word.text.lower() not in self.dicFreqWords:
                    countLowWord += 1
        self.numRareWord = countRareWord
        self.numLowWord = countLowWord          

    def sentenceComplexity(self):
        """
        Function that calculate the complexity at sentence level.
        """
        numComplexSentence=0
        for sentence in self.sentences:
            verb = False
            cont = 0
            for token in sentence:
                if token.pos_ == "VERB":
                    if verb:
                        verb= False
                        cont+=1
                    else:
                        verb = True
                else:
                    verb= False

            if cont > 0:
                numComplexSentence += 1
        self.numComplexSents = numComplexSentence
        self.avgLenSentence = self.numWords / self.numContentSentences
        self.complexSentence = self.numComplexSents / self.numContentSentences
        self.sentenceComplexityIndex = (self.avgLenSentence+self.complexSentence)/2
        

    def countSyllabes(self, text):
        """
        Function that count the syllabes of a text.

        Args:
            text: a string with the text to analyze.
        """
        t = re.sub(r'y([aáeéiíoóuú])', '\\1', text.lower())
        t = re.sub(r'[aáeéioóu][iuy]', 'A', t.lower())
        t = re.sub(r'[iu][aáeyéioóu]', 'A', t).lower()
        t = re.sub(r'[aáeéiíoóuúy]', 'A', t)
        return len(t.split('A'))-1

    def treeHeight(self,root, cont):
        if not list(root.children):
            return 1
        else:
            cont+=1
            if cont == 320:
              return 320
            return 1 + max(self.treeHeight(x, cont) for x in root.children)

    def embeddingDepth(self):
        """
        Function that calculate the depth of the embedding of a text.
        """
        roots = [sent.root for sent in self.sentences]
        max_list = []
        max_list = [self.treeHeight(root,0) for root in roots]
        
        self.max_max_list = max(max_list)
        self.min_max_list = min(max_list)
        self.mean_max_list = sum(max_list)/(len(max_list))
        
        return self.max_max_list,  self.min_max_list, self.mean_max_list

    def ageReadability(self):
        """
        Function that calculate the age readability of a text.
        """
        self.solReadability = -2.51 + 0.74*(3.1291+1.0430*math.sqrt(self.numWords3Syllabes*(30/self.numSentences)))
        self.minAge = 0.2495* (self.numWords/self.numSentences) + 6.4763*(self.numSyllabes/self.numWords) - 7.1395
        self.crawford = -20.5*(self.numSentences/self.numWords)+4.9*(self.numSyllabes/self.numWords)-3.407
        


