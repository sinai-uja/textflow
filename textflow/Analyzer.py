import os
from typing import Optional
from textflow.Sequence import Sequence

class Analyzer:
    def __init__(self, function, isMetadata: Optional[bool] = False):
        """Creates an analyzer from an input object.

        Args:
            function: the function of the analyzer like count word, files...
            isMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        self.function = function
        self.isMetadata = isMetadata


    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = None, analyzeMetadata: Optional[bool] = False): #TODO
        """Analyze a sequence

        Args:
            sequence: the Sequence we want to analyze
            levelOfAnalyzer: the path of the sequence level to analyze
            levelOfResult: the path of the sequence level to store the result. (Podemos querer analizar los tokens pero almacenarlo a nivel de oracion)
            tag: the label to store the analysis resut
            analyzeMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        ruta = levelOfAnalyzer.split("/")
        children = [sequence.children]
        results=[]
        for r in ruta:
            for child in children:
                if r in child:
                    if r == ruta[-1]:
                        results.extend(child[r])
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")
        analyze = self.function(results)
        print(analyze)
        ruta = levelOfResult.split("/")
        children = [sequence.children]
        for r in ruta:
            for child in children: 
                if r in child:
                    if r == ruta[-1]:
                        #child[r] es una lista de sequencias
                        #child[r].children[tag]=[]
                        #print(child[r])
                        for seq in child[r]:
                            print(seq)
                            seq.children[tag] = analyze.pop(0) 
                            print(seq.children)
                            '''child[r][chi].children[tag]=[]
                            child[r][chi].children[tag].append(analyze.pop(chi))'''
                            pass
                        #results.extend(child[r])
                        #pass
                        #Aqui ya dbo almacenar los resultados

                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")
        pass

        
        
    

        