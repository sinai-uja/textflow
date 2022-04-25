import os
from typing import Optional
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import SpaceTokenizer
from nltk.tokenize import WordPunctTokenizer


class SequenceIterator:
    def __init__(self, children):
        """
        Creates a sequenceIterator from a Sequence.
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


_VALID_FORMATS = ["directory","string", "text", "token", None]

class Sequence:
    """Summary of class here.

    Longer class information...
    Longer class information...

    Attributes:
        id: ...
        text: ...
        sequences: ...
    """
    def __init__(self, format: Optional[str] = None, src: Optional[object] = None, tokenizer: Optional[object] = None ):
        """Creates a sequence from an input object.

        Args:
            format: A string containing the input data's type.
            src: An object representing the input data. It can be a string for a
            string format or a file path for a text format.
        Raises:
            ValueError: If the format is wrong.    
        """ 
        if format not in _VALID_FORMATS:
            raise ValueError(
                f"{format} is not a valid format. Valid formats: {_VALID_FORMATS}"
            )
        if tokenizer == None:
            tokenizer = WhitespaceTokenizer()
        
        self.format = format
        self.children = {}
        self.metadata = {"text": " "}
        if format == "token":
            if not isinstance(src, str):
                raise ValueError(f"{src} is not an instance of token")
            self.metadata["text"] = src
        if format == "string":
            self.initFromString(src,"tokens","token",tokenizer)
        if format == "text":
            self.initFromDocument(src,"tokens","token", tokenizer)
        if format == "directory":
            self.initFromDirectory(src,"directory","files",tokenizer)

    def initFromDirectory(self, directory, labelDirectory, labelFile, tokenizer):
        '''
        Initialize a Sequence from a directory 
        Args:
            directory: the path of a directory as string
            labelDirectory: the name of the children dictionary entry for the subpaths
            labelFile: the name of the children dictionary entry for the files
        '''
        #print(os.path.abspath((os.getcwd())))
        self.format = "directory"
        self.metadata["nameFiles"] = []
        self.metadata["directoriesPath"] = []
        contenido = os.listdir(directory)
        #print(contenido)
        for file in contenido:
            #print(file)
            if os.path.isfile(directory+"/"+file):
                self.metadata["nameFiles"].append(file)
                if labelFile in self.children:
                    self.children[labelFile].append(Sequence("text", directory+"/"+file ))
                else:
                    self.children[labelFile]= [Sequence("text", directory+"/"+file)]
            else:
                self.metadata["directoriesPath"].append(directory+"/"+file)
                if labelDirectory in self.children:
                    self.children[labelDirectory].append(Sequence("directory", directory+"/"+file,tokenizer ))
                else:
                    self.children[labelDirectory]= [Sequence("directory", directory+"/"+file, tokenizer)]
        

    def initFromDocument(self, documentPath, labelSubSequence, formatSubsequence, tokenizer):
        '''
        Initialize a Sequence from a document 
        Args:
            documentPath: the path of a document as string
            labelSubSequence: the name of the children dictionary entry for the subsequence as string
            formatSubSequence: the format of the subsequence in children dictionary entry as string
        '''
        self.format = "text"
        with open(documentPath, "r") as f:
            txt = f.read()
        self.children[labelSubSequence] = [Sequence(formatSubsequence,token_src) for token_src in tokenizer.tokenize(txt)]
        self.metadata["text"] = txt

    def initFromString(self, srcString, labelSubSequence, formatSubsequence, tokenizer):
        '''
        Initialize a Sequence from a string 
        Args:
            srcString: source string of the sequence
            labelSubSequence: the name of the children dictionary entry for the subsequence as string
            formatSubSequence: the format of the subsequence in children dictionary entry as string
        Raises:
            ValueError: If srcString isn't a string .
        '''
        if not isinstance(srcString, str):
            raise ValueError(f"{srcString} is not an instance of string")
        self.format = "string"
        self.children[labelSubSequence]= [Sequence(formatSubsequence,token_src) for token_src in tokenizer.tokenize(srcString)]
        self.metadata["text"]= srcString


    def __str__(self):
        '''
         Convert a Sequence to a string
        
        Returns:
           A string that contains the text of a Sequence  
        '''
        return str(self.metadata["text"])

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

    def __len__(self):
        '''
        Calculate the length of a Sequence
        The length of a Sequence is the length of the children.
        Returns:
            A number with the length of the Sequence
        '''
        return len(self.children)

    def __iter__(self):
        '''
        Iterate in a Sequence
        To do this, we iterates througth the children dictionary 
        Returns:
            A Sequence Iterator  
        '''
        return SequenceIterator(list(self.children.values()))
    
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

    def __eq__(self, other):
        if self.format == other.format and self.metadata == other.metadata and self.children == other.children:
            return True
        else:
            return False


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
        for r in ruta:
            for child in children:
                if r in child:
                    if r == ruta[-1]:
                        results.extend(child[r])
                    else:
                        children = [c.children for c in child[r]]
                else:
                    raise ValueError(f"Sequence level '{r}' not found in {child}")
        yield criteria(results)
        