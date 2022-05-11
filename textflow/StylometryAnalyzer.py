from typing import Optional

import spacy
import spacy.cli


class StylometryyAnalyzer: #TODO
    def __init__(self, lang = "es"):
        if lang == "es":
            spacy.cli.download("es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm")
        self.function = self.stylometry
        pass

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
        if levelOfResult == "":
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
                        raise ValueError(f"Sequence level '{r}' not found in {child}") 

    def stylometry(self):
        pass
