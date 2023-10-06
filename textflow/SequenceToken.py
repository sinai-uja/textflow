import os
from typing import Optional
from nltk.tokenize import WhitespaceTokenizer
from textflow.Sequence import Sequence


class SequenceToken (Sequence):
    """
    A class that provides methods to create a sequence from a token

    Attributes:
        format: a string with the origin format of a sequence.
        metadata: a dictionary with the metadata of a sequence.
        children: a dictionary with the subsequence of a sequence. 
    """


    def __init__(self,src,listLabel,listClasses,listTokenizer=None):
        '''
        Initialize a Sequence from a token

        Args:
            src: source string of the sequence
            listLabel: a list with different labels to create new levels in the children dictionary
            listClasses: a list with different classes that inicialize a sequence with sublevels
            listTokenizer: a list with the tokenizer to inicialize the different subsequences

        '''
        self.initializeSequence("token")
        self.metadata["text"] = src

    def initializeSequence(self, format):
        '''
        Initializes the attributes of a sequence.

        Args:
            format: a string with the origin format of the sequence.
        '''
        super().initializeSequence(format)

    def __str__(self):
        '''
         Convert a Sequence to a string
        
        Returns:
           A string that contains the text of a Sequence  
        '''
        return super().__str__()
    

    def __repr__(self):
        '''
        Convert a Sequence to a string
        
        Returns:
           A string with the formal representation of a Sequence  
        '''
        return super().__repr__()

    def __len__(self):
        '''
        Calculate the length of a Sequence.
        The length of a Sequence is the length of the children.

        Returns:
            A number with the length of the Sequence
        '''
        return super().__len__()

    def __iter__(self):
        '''
        Iterate in a Sequence
        To do this, we iterates througth the children dictionary 
        Returns:
            A Sequence Iterator  
        '''
        return super().__iter__()
    
    def __getitem__(self, idx):
        '''
        Get the value of a key from the dictionary of children 

        Args:
            idx: a string that represent the key of the children dictionary
                 or an integer that represent the position of the key in children dictionary keys 
        Returns:
            A List of Sequences 
        '''
        return super().__getitem__(idx)

    def __eq__(self, other):
        '''
        Check if a sequence it is the same that the current one.

        Args:
            other: a sequence to check if it is the same that the current one.
        Returns:
            True if the sequences are equals.
            False in others cases.
        '''
        return super().__eq__(other)

    def depth(self,dictionaryList: Optional[list] = None):
        '''
        Calculate the maximum depth of a Sequence

        Args:
            diccionaryList: the inicial list to calculate the depth.

        Returns:
            A tuple that contains a number (the depth of a Sequence) and a list (the route of the max depth) 
        '''
        return super().depth(dictionaryList)
    
    def filter(self, level, criteria):
        '''
        Filter the children of a Sequence according to a criteria

        Args:
            level: the route of the level as string, separating each level with "/" 
            criteria: the filter function

        Returns:
            A generator with the result of the filter
        '''
        return super().filter(level,criteria)
    
    def filterMetadata(self, level, criteria):
        '''
        Filter the children of a Sequence according to a criteria

        Args:
            level: the route of the level as string, separating each level with "/" 
            criteria: the filter function

        Returns:
            A generator with the result of the filter
        '''
        return super().filterMetadata(level,criteria)

    def toDF(self,level):#, level, criteria):
        '''
        Filter the children of a Sequence according to a criteria

        Args:
            level: the route of the level as string, separating each level with "/" 
            criteria: the filter function

        Returns:
            A generator with the result of the filter
        '''
        return super().toDF(level)
