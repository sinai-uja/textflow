from typing import Optional
from abc import ABC, abstractmethod
import pandas as pd
from collections import defaultdict

class SequenceIterator:
    """
    A class that provides methods to iterate over the children of a sequence

    Attributes:
        idx: an integer with the position of the iterator.
        children: a dictionary with the subsequence of a sequence.  
    """

    def __init__(self, children):
        """
        Create a sequenceIterator from a Sequence.
        Args:
            children: A list with the values of the attribute children of a Sequence.
        """
        self.idx = 0
        self.children = children
    
    def __iter__(self):
        """

        Return:
            The sequence where the iterator is point.
        """
        return self

    def __next__(self):
        """
        Move the iterator to the next position.
        Return:
            children: A list with the values of the attribute children of a Sequence.
        """
        self.idx += 1
        try:
            return self.children[self.idx-1]
        except IndexError:
            self.idx = 0
            raise StopIteration


class Sequence(ABC):
    """
    Abstract class that provides methods to create a sequence from different formats

    Attributes:
        format: a string with the origin format of a sequence.
        metadata: a dictionary with the metadata of a sequence.
        children: a dictionary with the subsequence of a sequence. 
    """

    @abstractmethod
    def initializeSequence(self,format):
        '''
        Initializes the attributes of a sequence.

        Args:
            format: a string with the origin format of the sequence.
        '''
        self.format = format
        self.metadata={}
        self.children={}
        return self.format, self.metadata, self.children
   
    @abstractmethod
    def __str__(self):
        '''
         Convert a Sequence to a string
        
        Returns:
           A string that contains the text of a Sequence  
        '''
        return str(self.metadata["text"])
    
    @abstractmethod
    def __repr__(self):
        '''
        Convert a Sequence to a string
        
        Returns:
           A string with the formal representation of a Sequence  
        '''
        format = self.format
        return (
            "Sequence(\n"
            f"  format: {self.format}\n"
            f"  metadata: {str(self.metadata)}\n"
            f"  children: {str(self.children)}\n"
            ")"
        )

    @abstractmethod
    def __len__(self):
        '''
        Calculate the length of a Sequence.
        The length of a Sequence is the length of the children.

        Returns:
            A number with the length of the Sequence
        '''
        return len(self.children)

    @abstractmethod
    def __iter__(self):
        '''
        Iterate in a Sequence
        To do this, we iterates througth the children dictionary 
        Returns:
            A Sequence Iterator  
        '''
        return SequenceIterator(list(self.children.values()))
    
    @abstractmethod
    def __getitem__(self, idx):
        '''
        Get the value of a key from the dictionary of children 

        Args:
            idx: a string that represent the key of the children dictionary
                 or an integer that represent the position of the key in children dictionary keys 
        Returns:
            A List of Sequences 
        '''
        if isinstance(idx, str):  # Get src by string (e.g. seq["doc1"])
            if self.children:
                if idx in self.children: 
                    return self.children[idx]
            raise ValueError(f"Sequence id '{idx}' not found in {self.children.keys()}")
        elif isinstance(idx, int):  # Get src by int (e.g. seq[0])
            if abs(idx) >= len(self.children):
                raise IndexError(f"Sequence index '{idx}' out of range")

            if idx < 0:
                idx = len(self.children) + idx
            
            return list(self.children.values())[idx]
        else: # TODO: Should it support slices (e.g. [2:4])?
            raise ValueError(f"Sequence id '{idx}' not found in {self.children}")

    @abstractmethod
    def __eq__(self, other):
        '''
        Check if a sequence it is the same that the current one.

        Args:
            other: a sequence to check if it is the same that the current one.
        Returns:
            True if the sequences are equals.
            False in others cases.
        '''
        if self.format == other.format and self.metadata == other.metadata and self.children == other.children:
            return True
        else:
            return False

    @abstractmethod
    def depth(self,diccionaryList: Optional[list] = None):
        '''
        Calculate the maximum depth of a Sequence

        Args:
            diccionaryList: the inicial list to calculate the depth.

        Returns:
            A tuple that contains a number (the depth of a Sequence) and a list (the route of the max depth) 
        '''
        profMax = 0
        rutaMax = []
        if diccionaryList == None:
            diccionaryList = [self.children]
        for elemento in diccionaryList: #Recorre todos los elementos de la lista (diccionarios)
            for child in elemento: #Recorremos todas las claves del diccionario
                prof=0
                ruta=[child]
                if elemento[child] and isinstance(elemento[child], list):
                    listaDic = [seq.children for seq in elemento[child]]
                    depthChildren = self.depth(listaDic)
                    prof += depthChildren[0] + 1
                    ruta.extend(depthChildren[1])
                    if profMax < prof :
                        profMax = prof 
                        rutaMax = ruta
        return (profMax, rutaMax)
    
    @abstractmethod
    def filter(self, level, criteria):
        '''
        Filter the children of a Sequence according to a criteria

        Args:
            level: the route of the level as string, separating each level with "/" 
            criteria: the filter function

        Returns:
            A generator with the result of the filter
        '''
        ruta = level.split("/")
        children = [self.children]
        results=[]
        for idx, r in enumerate(ruta):
            for child in children:
                if r in child:
                    if r == ruta[-1] and idx == len(ruta)-1:
                        results.extend(child[r])
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")
        cont=0
        gen = criteria(results)
        for r in gen:
            yield gen[cont]
            cont+=1
    
    @abstractmethod
    def filterMetadata(self, level, criteria):
        '''
        Filter the children of a Sequence according to a criteria

        Args:
            level: the route of the level as string, separating each level with "/" 
            criteria: the filter function

        Returns:
            A generator with the result of the filter
        '''
        ruta = level.split("/")
        children = [self.children]
        metadata = [self.metadata]
        results=[]
        for idx, r in enumerate(ruta):
            if r == ruta[-1] and idx == len(ruta)-1:
                for m in metadata:
                    if r in m:
                        results.append(m[r])
            elif idx == len(ruta)-2 and True in [r in c.keys() for c in metadata]:
                aux=[]
                metadata=[]
                for c in metadata:
                    aux.extend(c[r])
                metadata.extend(aux)
            else:
                for child in children:
                    if r in child:
                        children = [c.children for c in child[r]]
                        metadata = [c.metadata for c in child[r]]
                    else:
                        raise ValueError(f"Sequence level '{r}' not found in {child}")
        cont=0
        gen = criteria(results)
        for r in gen:
            yield gen[cont]
            cont+=1
    
    @abstractmethod
    def toDF(self, level= "metadata"):
        '''
        Convert a Sequence to a pandas DataFrame

        #Args:
        #    level: the route of the level as string, separating each level with "/" 

        Returns:
            A pandas DataFrame with the sequence information
        '''
        path = level.split("/")
        children = [self.children]
        metadata = [self.metadata]
        columns = []
        values = []
        for idx, p in enumerate(path):
            print(idx,p)
            if p == "metadata":
                print("Es metadata",metadata)
                for metadataDic in metadata:
                    for m in metadataDic:
                        columns.append(m+str(idx))
                        values.append({str(m)+str(idx):metadataDic[m]})
            elif p == "children":
                childrenAux=[]
                for child in children: #Accedemos a los hijos
                    for ch in child: #Cada hijo tiene un diccionario de valores
                        #print(ch,child[ch])
                        columns.append(ch+str(idx))
                        auxDic={}
                        for c in child[ch]: #Dentro de cada diccionario de valores tenemos más sequencias
                            #print(c.metadata)
                            #print(c.children)
                            childrenAux.append(c.children)
                            for metadataKey in c.metadata: #Cada Sequencia tiene sus metadatos, como todas las sequencias de este nivel pertenecen al mismo, todas deberían tener los mismos metadatos
                                #print(metadataKey)
                                if metadataKey not in auxDic:
                                    auxDic[metadataKey]=[c.metadata[metadataKey]]
                                else:
                                    auxDic[metadataKey].append(c.metadata[metadataKey])                        
                        values.append({str(ch)+str(idx):auxDic})
                children = childrenAux

        finalColumns = []
        finalRows = {}
        for value in values: #Recorremos la lista de valores (diccionarios)
            for v in value: # Para cada clave en el diccionario
                print(v,type(value))
                if type(value[v]) == dict:
                    for keyValue in value[v]:
                        if v+keyValue not in finalColumns:
                            finalColumns.append(v+keyValue)
                            if type(value[v][keyValue]) == list:
                                finalRows[v+keyValue]=value[v][keyValue]
                            else:
                                finalRows[v+keyValue]=[value[v][keyValue]]
                        else:
                            if type(value[v][keyValue]) == list and finalRows[v+keyValue][0] != list: #Solo se ejecutara una vez despues de haber creado el dataset
                                newList=[]
                                for element in finalRows[v+keyValue]:
                                    newList.append(element)
                                finalRows[v+keyValue]=[newList,value[v][keyValue]]

                            else:
                                finalRows[v+keyValue].append(value[v][keyValue])
                else:
                    if len(value[v]) != 0:
                        if v not in finalColumns:
                            finalColumns.append(v)
                            finalRows[v]=value[v]
        print(finalRows)
        df = pd.DataFrame(finalRows)
        return df
