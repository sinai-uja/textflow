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

class ComplexityAnalyzer(Analyzer):
    def __init__(self, rutaArchivoCrea = creaPath,lang = "es"):
        """Creates an analyzer from an input object.

        Args:
            function: the function of the analyzer like count word, files...
            isMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        if lang == "es":
            spacy.cli.download("es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm")
            #Vamos a cargar CREA:
            self.dicFreqWords=self.read(rutaArchivoCrea)

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
        super().analyze(self.complexity,sequence, tag, levelOfAnalyzer, levelOfResult, True)
        '''if levelOfResult == "":
            analyzeResult = sequence.filterMetadata(levelOfAnalyzer,self.function)#TODO
            resultOfAnalisys= []
            for i in analyzeResult:
                resultOfAnalisys.append(i)
            sequence.metadata[tag] = resultOfAnalisys
        else:
            children = [sequence.children]
            ruta = levelOfResult.split("/")
            for r in ruta: #Para cada nivel de la ruta
                for child in children: #Miramos en todas las secuencias disponibles
                    if r in child: #Si dentro de la secuencia actual está r
                        if r == ruta[-1]:
                            for seq in child[r]:
                                analyzeResult = seq.filterMetadata(levelOfAnalyzer,self.function)  
                                resultOfAnalisys= []
                                for i in analyzeResult:
                                    resultOfAnalisys.append(i)
                                seq.metadata[tag] = resultOfAnalisys                           
                        else:
                            children = [c.children for c in child[r]]
                    else:
                        raise ValueError(f"Sequence level '{r}' not found in {child}") '''


    def read(self,fichero):
        with open(fichero,'r',encoding='latin-1') as file:
            next(file)
            lines = file.readlines()
            
        freqWordsCrea ={}    
        for l in lines[:-2]:
            words = l.strip().split()
            freqWordsCrea[words[1]] = float(words[2].replace(',',''))
        return freqWordsCrea



    def complexity(self, arrayText):
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
        #Simple metrics son los signos de puntuación, el numero de frases, el numero de frases con contenido...
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
        self.readabilityFH = 206.84 - 0.60*(self.numSyllabes/self.numWords) - 1.02*(self.numWords/self.numSentences)
        self.perspicuityIFSZ = 206.835 - ((62.3*self.numSyllabes)/self.numWords) - (self.numWords/self.numSentences)
        
        numLetters = 0
        listLenLetters =[]
        for token in doc:
            if token.text.isalpha(): #Si es una palabra
                numLetters += len(token.text)
                listLenLetters.append(len(token.text))
        
        avgLettersWords = numLetters/self.numWords
        listLenLetters = np.array(listLenLetters)
        
        self.poliniComprensibility = 95.2 - (9.7 * avgLettersWords) - ((0.35*self.numWords)/self.numSentences)        
        self.muLegibility = (self.numWords/(self.numWords-1))*(avgLettersWords/listLenLetters.var())*100
        
    def lexicalIndex(self):
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
        self.autoReadabilityIndex = 4.71 * self.numChars / self.numWords + 0.5 * self.numWords/self.numContentSentences
        self.spauldingScore = 1.609*(self.numWords / self.numContentSentences) + 331.8* (self.numRareWord /self.numWords) + 22.0

    def countRareAndLowWord(self):
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
        roots = [sent.root for sent in self.sentences]
        max_list = []
        max_list = [self.treeHeight(root,0) for root in roots]
        
        self.max_max_list = max(max_list)
        self.min_max_list = min(max_list)
        self.mean_max_list = sum(max_list)/(len(max_list))
        
        return self.max_max_list,  self.min_max_list, self.mean_max_list

    def ageReadability(self):
        self.solReadability = -2.51 + 0.74*(3.1291+1.0430*math.sqrt(self.numWords3Syllabes*(30/self.numSentences)))
        self.minAge = 0.2495* (self.numWords/self.numSentences) + 6.4763*(self.numSyllabes/self.numWords) - 7.1395
        self.crawford = -20.5*(self.numSentences/self.numWords)+4.9*(self.numSyllabes/self.numWords)-3.407
        pass


