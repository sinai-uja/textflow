
import spacy
import spacy.cli
from typing import Optional
from textflow.Analyzer import Analyzer

spacy.cli.download("es_core_news_sm")

class NERAnalyzer(Analyzer):
    """
    A class that provides methods to analyze the NER of the text of a sequence.

    Attributes:
        nlp: a model of language.
        textNER:the text with the entities instead of the words.
        dicEntidades:a dictionary with the entities.
        dicEntidadesFrecuencia: a dictionary with the frequence of the different entities.
    """

    def __init__(self, nlp = spacy.load("es_core_news_sm")):
        """
        Create a NER analyzer from an input object.

        Args:
            nlp: a model of language.
        """
        self.nlp = nlp

    def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
        """
        Analyze a sequence with a NER function.

        Args:
            sequence: the Sequence we want to analyze.
            tag: the label to store the analysis result.
            levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
            levelOfResult: the path of the sequence level to store the result. 
        """
        super().analyze(self.nerAnalyzer,sequence, tag, levelOfAnalyzer, levelOfResult, True)


    def nerAnalyzer(self, arrayText):
        '''
        Function that get the NER of a list of texts.

        Args:
            arrayText: list that contains the texts that we want to analyze
        
        Returns:
            A list with the dictionaries. Each dictionary contains the result
            of the analysis of the corresponding text.
        '''
        arrayResult = []
        for text in arrayText:
            self.freqNer(text)
            ner={
                "srcNER" : self.textNER,
                "entidades" : self.dicEntidades,
                "freqEntidades" : self.dicEntidadesFrecuencia
            }
            arrayResult.append(ner)
        return arrayResult

    def freqNer(self,text):
        """
        Function that counts the number of the different categories of NER that appear in the text, their frequency and
        changes the corresponding word by its NER category

        Args:
            text: the text that we want to analyze
        """
        self.dicEntidades= {}
        self.dicEntidadesFrecuencia = {}
        doc = self.nlp(text)
        textner=[]
        for i in range(len(doc)):
            if doc[i].ent_type_ != '':
                textner.append(doc[i].ent_type_)
            else:
                textner.append(doc[i].text)
        self.textNER = " ".join(textner) 
        for ent in doc.ents:
            #Guardamos el diccionario obtenido para la categoria de la palabra (si este existe)
            dicPalabras = self.dicEntidades.get(ent.label_)
            
            #Si hay un diccionario, es decir es una categoría que ha aparecido previamente
            if dicPalabras != None:
                #Aumentamos la frecuencia de aparición en esta categoría
                self.dicEntidadesFrecuencia[ent.label_] += 1
                #introducimos en el diccionario la palabra
                if ent.text.lower() in dicPalabras:
                    dicPalabras[ent.text.lower()] += 1
                else:
                    dicPalabras[ent.text.lower()] = 1

            #Si es igual de none, no tenemos esa categoria
            else:
                #Creamos el diccionario para esta categoria
                palabrasFrecuencia ={}
                #Insertamos la palabra actual en el diccionario
                palabrasFrecuencia[ent.text.lower()] = 1
                #Insertamos el diccionario dentro del diccionario de categorias para la categoria asociada
                self.dicEntidades[ent.label_] = palabrasFrecuencia
                #Ponemos a uno la frecuencia de aparición para esa categoría de entidad
                self.dicEntidadesFrecuencia[ent.label_] = 1



            
                                
                                



