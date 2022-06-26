# TextFlow class

This class provides methods to create a sequence from directories, documents, strings… To calculate different metrics from these sequences. It consists in a Python library for calculating different metrics from plain texts.

# Files

 - **[INSTALL.md](https://gitlab.ujaen.es/jcollado/textflow/blob/master/INSTALL.md):** A guide to make this project work on your local environment.

### ./textFlow

- **[Analyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/Analyzer.py):** This module provides an abstract class with methods for the calculation of different metrics on a sequence.

- **[ComplexityAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/ComplexityAnalyzer.py):** This module provides a class methods for the calculation of different complexity metrics on a sequence. This class inherits from Analyzer.py

- **[CREA_5000.txt](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/Crea-5000.txt):** A dataset of 5000 spanish words ordered by its absolute frequency.

- **[EmotionAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/EmotionAnalyzer.py):** This module provides a class methods for the calculation of the emotions on a sequence. This class inherits from Analyzer.py

- **[LemmaAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/LemmaAnalyzer.py):** This module provides a class methods for the calculation of different lemma metrics on a sequence. This class inherits from Analyzer.py

- **[POSAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/POSAnalyzer.py):** This module provides a class methods for the calculation of different Part-of-speech metrics on a sequence. This class inherits from Analyzer.py

- **[PolarityAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/PolarityAnalyzer.py):** This module provides a class methods for the calculation of the polarity on a sequence. This class inherits from Analyzer.py

- **[Sequence.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/Sequence.py):** This module provides an abstract class with methods for creating sequences from dfferent sources. A sequence contains 2 dictionaries, one for metadata and other for the subsequence of this sequence.

- **[SequenceDirectory.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceDirectory.py):** This module provides methods for creating a sequence from a directory.

- **[SequenceFile.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceFile.py):** This module provides methods for creating a sequence from a file or a document.

- **[SequenceString.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceString.py):** This module provides methods for creating a sequence from a string.

- **[SequenceToken.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/SequenceToken.py):** This module provides methods for creating a sequence from a token.

- **[StylometryAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/StylometryAnalyzer.py):** This module provides a class methods for the calculation of different stylometry metrics on a sequence. This class inherits from Analyzer.py

- **[VolumetryAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/VolumetryAnalyzer.py):** This module provides a class methods for the calculation of different volumetry metrics on a sequence. This class inherits from Analyzer.py

- **[IronityAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/IronityAnalyzer.py):** This module provides a class methods for the calculation of the ironity on a sequence. This class inherits from Analyzer.py

- **[NERAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/NERAnalyzer.py):** This module provides a class methods for the search of different NER on a sequence. This class inherits from Analyzer.py

- **[NGramsAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/NGramsAnalyzer.py):** This module provides a class methods for the calculation of n-grams and their frequence on a sequence. This class inherits from Analyzer.py

- **[EmojiAnalyzer.py](https://gitlab.ujaen.es/jcollado/textflow/blob/master/textflow/EmojiAnalyzer.py.py):** This module provides a class methods for the calculation of different emojis metrics on a sequence. This class inherits from Analyzer.py

**Note:** All of the analyzers implemented by default are applied to plain text.

### ./examples

- **[example_text.txt](https://gitlab.ujaen.es/jcollado/textflow/blob/master/Examples/ExampleDirectory/Documento%20sin%20t%C3%ADtulo.txt):** Simple .txt file to test the library.

- **[example.ipynb](https://gitlab.ujaen.es/jcollado/textflow/blob/master/Examples/Example.ipynb):** Colab notebook that shows how to use the different methods of textFlow.

# Metrics

In this section, we introduce the different metrics offered in this Python library. These metrics are returned by the corresponding analyzer and store in the corresponding dictionary (metadata or children) of a sequence.


- **Volumetry:** Here it calculates the number of words, number of unique words, number of characters and average word length for text.

- **Lemmas:** Number of the lemmas that are uniques in the text, a list of lemma words and the average lemma length for text.

- **Part-of-speech (POS)**: a list with the POS of the text and the frequency of the different POS labels.

- **Complexity:** Number of sentences, complex sentences, punctuation marks, words, rare words,chars, syllabes, average sentence length, index of low frequency words, index of lexical distribution, lexical complexity, spaulding score, index of sentence complexity, auto readability, readability of Fernandez-Huerta, perspicuity of Flesh-Szigriszt , polaini compressibility, mu legibility, SOL readability,some indicators of age (min age, crawford) and different indicators of the embeddings depth.

- **Stylometry:** Number of different words, different lexical index (TTR,RTT, Herdan, Mass, Somers, Dugast, Honore), frequency of stopwords, frequency of punctuation marks, frequency of words.

- **Polarity:** Polarity score of a text.

- **Emotions:** Emotions score of a text.

- **Emojis:** Number of emojis of the text, their frequence and the text with the words of the emojis instead of the emoji.

- **NER:** the frequence of different entities, the entities grouped by each category and the text with entities instead of the words.

- **N-Grams:** the different n-grams of the text and their frequence.

- **Ironity:** Ironity score of a text.

# Dependencies

- **ComplexityAnalyzer.py, POSAnalyzer, LemmaAnalyzer and NERAnalyzer:** In these classes, spacy is used to calculate the different metrics of the analyzers. If do you want to use other package, you should implements the methods nlp, sents, pos_, lemma_ and text.

- **IronityAnalizer.py, EmotionAnalyzer.py and PolarityAnalyzer.py:** These classes use models and pipelines of transformers, you can use different models to inference the emotion or the polarity of a text.

- **EmojiAnalizer.py:** This class use emoji library.

# How to create a Sequence?


If you want to create a class to initialize a sequence from a file and there is no class for it, you can create your own. We must create a class that inherits from Sequence.py and then create the following functions:
        
- **initializeSequence**:

        def initializeSequence(self, format):
            '''
            Initializes the attributes of a sequence.

            Args:
                format: a string with the origin format of the sequence.
            '''
            super().initializeSequence(format)

- **__str__**:

        def __str__(self):
            '''
            Convert a Sequence to a string
            
            Returns:
            A string that contains the text of a Sequence  
            '''
            return super().__str__()
    
- **__repr__**:

        def __repr__(self):
            '''
            Convert a Sequence to a string
            
            Returns:
            A string with the formal representation of a Sequence  
            '''
            return super().__repr__()

- **__len__**:

        def __len__(self):
            '''
            Calculate the length of a Sequence.
            The length of a Sequence is the length of the children.

            Returns:
                A number with the length of the Sequence
            '''
            return super().__len__()

- **__iter__**:

        def __iter__(self):
            '''
            Iterate in a Sequence
            To do this, we iterates througth the children dictionary 
            Returns:
                A Sequence Iterator  
            '''
            return super().__iter__()

- **__getitem__**:

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

- **__eq__**:

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

- **depth**:

        def depth(self,dictionaryList: Optional[list] = None):
            '''
            Calculate the maximum depth of a Sequence

            Args:
                diccionaryList: the inicial list to calculate the depth.

            Returns:
                A tuple that contains a number (the depth of a Sequence) and a list (the route of the max depth) 
            '''
            return super().depth(dictionaryList)

- **filter**:

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

- **filterMetadata**:

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

All of these functions are necesary for a sequence to function correctly, but the most important function isn't implemented. 
Let's see how to implement it.

- **__init__**:

First, we have to call self.initializeSequence("format"), because this function initialize the metadata, and children dictionary of a Sequence and put the format of a sequence. Then, we have to think about the metadata that we will have and how the children sequences are going to be built in that initializer. Finally,we can create the sequences down to the lowest level by calling other sequence initializers and what labels they will have in the children dictionary. 

An example of a new stream initializer for a directory might look like this: 
        

        def __init__(self,src,listLabel = ["directories","files","tokens"],listClasses=[SequenceFile,SequenceString],listTokenizer=[WhitespaceTokenizer()]):
            '''
            Initialize a Sequence from a directory path

            By default, create subsequences for any directories and files in the source directory 
            and for each file, create subsequence, splitting the text of the file into words.

            Args:
                src: the path of the directory
                listLabel: a list with different labels to create new levels in the children dictionary
                listClasses: a list with different classes that inicialize a sequence with sublevels
                listTokenizer: a list with the tokenizer to inicialize the different subsequences

            '''
            # Initializes the attributes of a sequence.
            self.initializeSequence("directory") 

            # Create the metadata and children of a Sequence
            self.metadata["nameFiles"] = []
            self.metadata["directoriesPath"] = []
            if not listTokenizer or listTokenizer == None:
                    listTokenizer = [WhitespaceTokenizer()]
            contenido = os.listdir(src)
            for file in contenido:
                if os.path.isfile(src+"/"+file):
                    self.metadata["nameFiles"].append(file)
                    if listLabel and listClasses:
                        if listLabel[1] in self.children:
                        #Create a sublevel of sequence
                            self.children[listLabel[1]].append(listClasses[0](src+"/"+file,listLabel[1:],listClasses[1:],listTokenizer[1:])) 
                        else:
                        #Create a sublevel of sequence
                            self.children[listLabel[1]] = [listClasses[0](src+"/"+file,listLabel[1:],listClasses[1:],listTokenizer[1:])]
                
                else:
                    self.metadata["directoriesPath"].append(src+"/"+file)
                    if listLabel[0] in self.children:
                        self.children[listLabel[0]].append(SequenceDirectory(src+"/"+file,listLabel,listClasses,listTokenizer ))
                    else:
                        self.children[listLabel[0]] = [SequenceDirectory(src+"/"+file,listLabel,listClasses,listTokenizer)]


# How to create an Analyzer?

Create an analyzer is more easy than a sequence. The steps to create an analyzer are:

1. Create a class that inherits of Analyzer.py
2. Create the __init__ function of this class with the params to configurate this class
3. Create a function that analyze a list of thigs that we want to analyze. 

    + For example, we want to create an analyzer of text, so we need a function that receives a list of texts and applies what it needs to each text. 

    + We can see an example of a ironityAnalyzer:


        def ironity(self, arrayText):
            """
            Function that analyzes the ironity of a list of texts.

            Args:
                arrayText: list that contains the texts that we want to analyze
            Returns:
                A list with the dictionaries. Each dictionary contains the result
                of the analysis of the corresponding text.
            """
            arrayResults =[]
            for text in arrayText:
                prediction = self.ironityClassifier(text[:self.maxEmbeding])
                arrayResults.append(prediction)
            return arrayResults


4. Create the analyze function: 

    + Replace self.analyzeFunction with the analyzer function that we have implemented in point 3
    + The parameter "True" of the super().analyze is because this analyzer is a metadata analyzer. If do you want to create an analyzer of sequence, this parameter must be "False"



        def analyze(self, sequence, tag, levelOfAnalyzer, levelOfResult:Optional[str] = ""): 
                """
                Analyze a sequence with a ironic function.

                Args:
                    sequence: the Sequence we want to analyze.
                    tag: the label to store the analysis result.
                    levelOfAnalyzer: the path of the sequence level to analyze inside of the result.
                    levelOfResult: the path of the sequence level to store the result.
                """
                super().analyze(self.analyzeFunction,sequence, tag, levelOfAnalyzer, levelOfResult, True)

