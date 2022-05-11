
import spacy
import spacy.cli
from typing import Optional
from textflow.Sequence import Sequence
#from transformers import pipeline


class Analyzer:
    def __init__(self, function, isMetadata: Optional[bool] = False,lang : Optional[str] = "es"):
        """Creates an analyzer from an input object.

        Args:
            function: the function of the analyzer like count word, files...
            isMetadata: boolean, if the result of the analyzer is stored in metadata (True) or in children(False)
        """
        if lang == "es":
            spacy.cli.download("es_core_news_sm")
            self.nlp = spacy.load("es_core_news_sm")
        elif lang == "en":
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        self.lang = lang
        self.function = function
        self.isMetadata = isMetadata


    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = "", analyzeMetadata: Optional[bool] = False): #TODO
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
            if analyzeMetadata:
                analyzeResult = sequence.filterMetadata(levelOfAnalyzer, self.function)                
                resultOfAnalisys= []
                for i in analyzeResult:
                    resultOfAnalisys.append(i)
                if isinstance(resultOfAnalisys[0], Sequence):
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
                                    analyzeResult = seq.filterMetadata(levelOfAnalyzer, self.function)
                                    '''for i in analyzeResult:
                                        resultOfAnalisys = i'''
                                    
                                    resultOfAnalisys= []
                                    for i in analyzeResult:
                                        resultOfAnalisys.append(i)
                                    if isinstance(resultOfAnalisys[0], Sequence):
                                        seq.children[tag] = resultOfAnalisys        
                                    else:
                                        seq.metadata[tag] = resultOfAnalisys

                                else:
                                    analyzeResult = seq.filter(levelOfAnalyzer, self.function)
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


    #La secuencia siempre debe tener un atributo texto para que este funcione
    #Contar el numero de palabras, numero de palabras unicas, numero de caracteres y numero medio de caracteres
    def volumetry(self,sequence,levelOfAnalyze): #TODO: Revisar
        children = [sequence.children]
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
                    raise ValueError(f"Sequence level '{r}' not found in {child}")

    def lemmas(self, sequence, levelOfAnalyze): #TODO: Revisar
        children = [sequence.children]
        ruta = levelOfAnalyze.split("/")
        for r in ruta: #Para cada nivel de la ruta
            for child in children: #Miramos en todas las secuencias disponibles
                if r in child: #Si dentro de la secuencia actual está r
                    if r == ruta[-1]:
                        for seq in child[r]:
                            if "text" not in seq.metadata:
                                raise ValueError(f"Level text not found in {seq.metadata.keys()}")
                            else:
                                sequenceLemmas = []
                                setLemmas = set()
                                lemma ={}
                                sumaLenLemmas=0
                                text = seq.metadata["text"]
                                doc= self.nlp(text)
                                for token in doc:
                                    if token.pos_ not in ["PUNCT", "SPACE", "SYM"]:
                                        sumaLenLemmas += len(token.lemma_)
                                        setLemmas.add(token.lemma_)
                                        s = Sequence("token",token.lemma_)
                                        sequenceLemmas.append(s)

                                lemma["uniqueLemmas"] = len(setLemmas)
                                lemma["avgLemmasLen"] = round(sumaLenLemmas/len(sequenceLemmas))
                                seq.metadata["lemmas"] = lemma
                                seq.children["lemmas"] = sequenceLemmas
                                
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")
    
    #Es necesario tener una etiqueta de token en children, si esta no existe, se creará
    def pos (self, sequence, levelOfAnalyze): #TODO: Revisar
        children = [sequence.children]
        ruta = levelOfAnalyze.split("/")
        for r in ruta: #Para cada nivel de la ruta
            for child in children: #Miramos en todas las secuencias disponibles
                if r in child: #Si dentro de la secuencia actual está r
                    if r == ruta[-1]:
                        for seq in child[r]:
                            if "text" not in seq.metadata:
                                raise ValueError("The sequence of the level {levelOfAnalyze} don't have atribute text")
                            else:
                                doc = self.nlp(seq.metadata["text"])
                                
                                if "tokens" not in seq.children:
                                    #Creamos uno
                                    pos=[]
                                    for token in doc:
                                        s = Sequence("token",token.text)
                                        s.metadata["pos"] = token.pos_ 
                                        pos.append(s)
                                    seq.children["tokens"] = pos
                                else:
                                    pos=[]
                                    for token in doc:
                                        pos.append(token.pos_)
                                    for seqToken in seq.children["tokens"]:
                                        seqToken.metadata["pos"] = pos.pop(0) 
                                
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")
'''
    def polaridad(self, sequence, levelOfAnalyze):
        #https://huggingface.co/finiteautomata/beto-sentiment-analysis
        if self.lang == "es":
            polarityClassifier = pipeline("text-classification",model='finiteautomata/beto-sentiment-analysis', return_all_scores=True)
        elif self.lang == "en":
            polarityClassifier = pipeline("text-classification",model='finiteautomata/bertweet-base-sentiment-analysis', return_all_scores=True)

        children = [sequence.children]
        ruta = levelOfAnalyze.split("/")
        for r in ruta: #Para cada nivel de la ruta
            for child in children: #Miramos en todas las secuencias disponibles
                if r in child: #Si dentro de la secuencia actual está r
                    if r == ruta[-1]:
                        for seq in child[r]:
                            if "text" not in seq.metadata:
                                raise ValueError(f"Level text not found in {seq.metadata.keys()}")
                            else:
                                prediction = polarityClassifier(seq.metadata["text"])
                                seq.metadata["polarity"] = prediction
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}") 
        pass

    def emotions(self, sequence, levelOfAnalyze):
        if self.lang == "es":
            emotionsClassifier = pipeline("text-classification",model='pysentimiento/robertuito-emotion-analysis', return_all_scores=True)
        elif self.lang == "en":
            emotionsClassifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

        children = [sequence.children]
        ruta = levelOfAnalyze.split("/")
        for r in ruta: #Para cada nivel de la ruta
            for child in children: #Miramos en todas las secuencias disponibles
                if r in child: #Si dentro de la secuencia actual está r
                    if r == ruta[-1]:
                        for seq in child[r]:
                            if "text" not in seq.metadata:
                                raise ValueError(f"Level text not found in {seq.metadata.keys()}")
                            else:
                                prediction = emotionsClassifier(seq.metadata["text"])
                                seq.metadata["emotions"] = prediction
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")'''