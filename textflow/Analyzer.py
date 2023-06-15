
import spacy
import spacy.cli
from typing import Optional
from textflow.Sequence import Sequence
from abc import ABC, abstractmethod


class Analyzer(ABC):
    """
    Abstract class that provides methods to analyze sequences
    """

    @abstractmethod
    def analyze(self, functionAnalyzer,sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = "", analyzeMetadata: Optional[bool] = False): #TODO
        """
        Abstract Class that analyze a sequence.

        Args:
            functionAnalyzer: the function of the analyzer.
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis resut.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result. 
            analyzeMetadata: boolean, if the result of the analyzer is applied in metadata (True) or in children(False).

        Raises:
            ValueError if the levelOfResult is incorrect
        """
        if levelOfResult == "":
            if analyzeMetadata:
                analyzeResult = sequence.filterMetadata(levelOfAnalyzer, functionAnalyzer)                
                resultOfAnalisys= []
                for i in analyzeResult:
                    resultOfAnalisys.append(i)
                if len(resultOfAnalisys) >0 and isinstance(resultOfAnalisys[0], Sequence):
                    sequence.children[tag] = resultOfAnalisys        
                else:
                    sequence.metadata[tag] = resultOfAnalisys
        else:
            children = [sequence.children]
            ruta = levelOfResult.split("/")
            for r in ruta: #Para cada nivel de la ruta
                for child in children: #Miramos en todas las secuencias disponibles
                    if r in child: #Si dentro de la secuencia actual está r
                        if r == ruta[-1]:
                            for seq in child[r]:
                                if analyzeMetadata:
                                    analyzeResult = seq.filterMetadata(levelOfAnalyzer, functionAnalyzer)
                                    
                                    resultOfAnalisys= []
                                    for i in analyzeResult:
                                        resultOfAnalisys.append(i)
                                    if isinstance(resultOfAnalisys[0], Sequence):
                                        seq.children[tag] = resultOfAnalisys        
                                    else:
                                        seq.metadata[tag] = resultOfAnalisys

                                else:
                                    analyzeResult = seq.filter(levelOfAnalyzer, functionAnalyzer)
                                    for i in analyzeResult:
                                        resultOfAnalisys = i
                                    if isinstance(resultOfAnalisys[0], Sequence):
                                        seq.children[tag] = resultOfAnalisys        
                                    else:
                                        seq.metadata[tag] = resultOfAnalisys
                            
                        else:
                            children = [c.children for c in child[r]]
                    else:
                        raise ValueError(f"Sequence level '{r}' not found in {child}")


    

    