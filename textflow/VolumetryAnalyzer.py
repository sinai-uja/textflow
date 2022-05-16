from typing import Optional
from textflow.Sequence import Sequence
from nltk.tokenize import WhitespaceTokenizer
from textflow.Analyzer import Analyzer

class VolumetryAnalyzer(Analyzer):
    def __init__(self, tokenizer= WhitespaceTokenizer()):
        """Creates an analyzer from an input object.

        Args:
            function: the function of the analyzer like count word, files...
            isMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        self.tokenizer = tokenizer


    def volumetry(self, arrayText):
        arrayResults =[]
        for texts in arrayText:
            text = self.tokenizer.tokenize(texts)
            dicResults = { 
                "words" : len(text),
                "uniqueWords" : len(set(text)),
                "chars" : len(texts),
                "avgWordsLen" : round(len(texts) / len(text))         
            }
            arrayResults.append(dicResults)
        return arrayResults

    #La secuencia siempre debe tener un atributo texto(metadata) para que este funcione
    #Contar el numero de palabras, numero de palabras unicas, numero de caracteres y numero medio de caracteres
    def analyze(self,sequence,tag,levelOfAnalyzer,levelOfResult:Optional[str] = ""):
        super().analyze(self.volumetry,sequence, tag, levelOfAnalyzer, levelOfResult, True)
        '''children = [sequence.children]
        ruta = levelOfAnalyze.split("/")
        for r in ruta: #Para cada nivel de la ruta
            for child in children: #Miramos en todas las secuencias disponibles
                if r in child: #Si dentro de la secuencia actual está r
                    if r == ruta[-1]:
                        for seq in child[r]:
                            if "text" not in seq.metadata:
                                raise ValueError(f"Level text not found in {seq.metadata.keys()}")
                            else:
                                text = seq.metadata["text"].split(" ")
                            
                                volumetry= {
                                    "words" : len(text),
                                    "uniqueWords" : len(set(text)),
                                    "chars" : len(seq.metadata["text"]),
                                    "avgWordsLen" : round(volumetry["chars"] / volumetry["words"])
                                }

                                seq.metadata["volumetry"] = volumetry
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")'''

   